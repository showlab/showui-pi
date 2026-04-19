#!/usr/bin/env python
"""
Accelerate-based multi-GPU data-parallel training runner for LeRobot.

This script mirrors the training flow of `lerobot_train.py`, but wraps the
model/optimizer/dataloader with Hugging Face Accelerate to enable true
multi-process, multi-GPU training (one GPU per process).

Usage example (4 GPUs, bf16):

  PYTHONPATH=lerobot/src \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  accelerate launch --num_processes 4 --mixed_precision bf16 \
    -m lerobot.scripts.lerobot_train_accelerate \
    --policy.type=smolvla \
    --policy.push_to_hub=false \
    --policy.load_vlm_weights=true \
    --policy.resize_imgs_with_padding='[1024,576]' \
    --policy.chunk_size=21 --policy.n_action_steps=21 \
    --dataset.root=data/dex \
    --dataset.repo_id=dex --dataset.revision=v3.0 \
    --batch_size=4 --num_workers=8 \
    --steps=15000 --eval_freq=0

Notes
- This script does not modify the LeRobot library; it only orchestrates training with Accelerate.
- Each process handles a shard of data via DistributedSampler and synchronizes gradients.
"""

import logging
import math
import os
import time
import contextlib
import gc
import random
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing as mp

from accelerate import Accelerator

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.random_utils import set_seed
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.train_utils import (
    save_checkpoint,
    update_last_checkpoint,
    get_step_checkpoint_dir,
    load_training_state,
)

def _make_dataloader(dataset, batch_size: int, num_workers: int, accelerator: Accelerator) -> DataLoader:
    # NOTE: Do NOT add a DistributedSampler here. accelerator.prepare() will
    # automatically shard the dataloader across processes. Adding a manual
    # DistributedSampler causes double-sharding (each process sees 1/N^2 data).
    ctx_name = os.getenv("LEROBOT_MP_CTX", "spawn")
    mp_ctx = mp.get_context(ctx_name) if num_workers and num_workers > 0 else None
    return DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=(2 if num_workers and num_workers > 0 else None),
        persistent_workers=(num_workers > 0),
        timeout=(0 if num_workers and num_workers > 0 else 0),
        multiprocessing_context=mp_ctx,
    )


def _log(s: str, rank: int | None = None):
    # Only main process logs by default to reduce noise; override with LEROBOT_LOG_ALL_RANKS=1
    log_all = os.getenv("LEROBOT_LOG_ALL_RANKS", "0") == "1"
    if rank is not None and not log_all and rank != 0:
        return
    prefix = f"[rank {rank}] " if rank is not None else ""
    print(prefix + s, flush=True)


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    # Route factory INFO logs (dataset sizes, grounding mix) to stdout so
    # operators can verify data loading at training start. Without this, all
    # those logging.info calls are swallowed.
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Ensure presets (optimizer/scheduler from policy) and defaults (output_dir, job_name)
    cfg.validate()

    # Reduce CPU thread over-subscription that can look like a hang with multi-proc.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    # Helpful NCCL diagnostics if collectives hang
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

    accelerator = Accelerator()
    rank = accelerator.process_index
    world = accelerator.num_processes
    _log(f"Accelerator initialized (world={world}, device={accelerator.device})", rank)
    DEBUG_FETCH = os.getenv("LEROBOT_DEBUG_FETCH", "0") == "1"

    if cfg.seed is not None:
        # Make seed different across processes for data shuffling reproducibility
        set_seed(cfg.seed + accelerator.process_index)

    # Initialize Weights & Biases (main process only)
    wandb_logger = None
    if cfg.wandb.enable and cfg.wandb.project and accelerator.is_main_process:
        wandb_logger = WandBLogger(cfg)

    # Optional: load episode list from file.
    episodes_file = getattr(cfg.dataset, "episodes_file", None)
    if episodes_file:
        ds_root = str(cfg.dataset.root) if cfg.dataset.root is not None else None
        if ds_root is None:
            raise RuntimeError("episodes_file provided but cfg.dataset.root is None")
        import pathlib

        want = pathlib.Path(ds_root).resolve()
        eps: list[int] = []
        with open(episodes_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                repo, ep = line.split("\t")
                if pathlib.Path(repo).resolve() == want:
                    eps.append(int(ep))
        if len(eps) == 0:
            raise RuntimeError(f"episodes_file specified but no matching episodes for root {want}")
        cfg.dataset.episodes = sorted(set(eps))
        _log(f"[episodes_file] loaded {len(cfg.dataset.episodes)} episode indices from {episodes_file}", rank)

    # Dataset (meta used for processors)
    t0 = time.time()
    _log("Building dataset ...", rank)
    dataset = make_dataset(cfg)
    # Keep a reference to original dataset metadata even if we wrap dataset into a Subset later
    ds_meta = getattr(dataset, "meta", None)
    # Rebase episode dataset indices when training on a local subset without streaming.
    # Use actual per-episode counts from the local hf_dataset to avoid out-of-bounds.
    try:
        import datasets as _hfds  # type: ignore
        from collections import defaultdict

        has_hf = hasattr(dataset, "hf_dataset") and dataset.hf_dataset is not None
        has_meta_eps = ds_meta is not None and hasattr(ds_meta, "episodes") and ds_meta.episodes is not None
        if has_hf and has_meta_eps:
            eps_ds = ds_meta.episodes
            col_names = set(eps_ds.column_names)
            if not ("dataset_from_index" in col_names and "dataset_to_index" in col_names):
                raise RuntimeError("Episode metadata missing 'dataset_from_index'/'dataset_to_index' columns")

            # Read episode_index column from local hf_dataset and count present rows per episode
            ep_col = dataset.hf_dataset["episode_index"]
            first_pos: dict[int, int] = {}
            last_pos: dict[int, int] = {}
            any_found = False
            for i, v in enumerate(ep_col):
                any_found = True
                ep_i = int(v.item()) if hasattr(v, "item") else int(v)
                if ep_i not in first_pos:
                    first_pos[ep_i] = i
                last_pos[ep_i] = i
            if not any_found:
                raise RuntimeError("No frames found in local hf_dataset to compute episode boundaries.")

            # Build new from/to based on actual local row positions; keep untouched episodes unchanged
            from_col = list(eps_ds["dataset_from_index"])  # type: ignore
            to_col = list(eps_ds["dataset_to_index"])  # type: ignore
            new_from = from_col.copy()
            new_to = to_col.copy()

            for ep_idx in sorted(first_pos.keys()):
                new_from[ep_idx] = int(first_pos[ep_idx])
                new_to[ep_idx] = int(last_pos[ep_idx]) + 1

            data_dict = {name: eps_ds[name] for name in eps_ds.column_names}
            data_dict["dataset_from_index"] = new_from
            data_dict["dataset_to_index"] = new_to
            ds_meta.episodes = _hfds.Dataset.from_dict(data_dict)
            _log("[accelerate] Rebased episode indices using local row boundaries (map-style).", rank)
    except Exception as _e:  # noqa: BLE001
        _log(f"[accelerate] Note: could not rebase episode indices from local counts: {_e}", rank)

    # Strictly filter rows to the requested episodes if provided in cfg
    try:
        requested = getattr(cfg.dataset, "episodes", None)
        if requested is not None and len(requested) > 0:
            allowed = {int(e) for e in requested}
            eps_col = dataset.hf_dataset["episode_index"]  # type: ignore[attr-defined]
            keep: list[int] = []
            for i, v in enumerate(eps_col):
                vi = int(v.item()) if hasattr(v, "item") else int(v)
                if vi in allowed:
                    keep.append(i)
            if len(keep) == 0:
                _log("[accelerate] Warning: episodes provided, but none found locally; training would be empty.", rank)
            else:
                dataset = Subset(dataset, keep)
                _log(f"[accelerate] Filtered dataset rows by episodes: kept={len(keep)} of total={len(eps_col)}", rank)
    except Exception as _e:  # noqa: BLE001
        _log(f"[accelerate] Note: could not apply episode filter: {_e}", rank)
    # Optional debug wrapper for dataset __getitem__ timings
    if os.getenv("LEROBOT_DEBUG_DATA", "0") == "1":
        import types
        from torch.utils.data import Dataset as _TorchDataset  # type: ignore

        class _DebugDataset(_TorchDataset):  # type: ignore
            def __init__(self, inner, dbg_rank: int, log_every: int = 0, max_items: int = 16):
                self.inner = inner
                self.rank = dbg_rank
                self.log_every = log_every
                self.max_items = max_items
                self._seen = 0

            def __len__(self):
                return len(self.inner)

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def __getitem__(self, idx):
                t0 = time.time()
                item = self.inner[idx]
                dt = time.time() - t0
                if self._seen < self.max_items or (self.log_every and idx % self.log_every == 0):
                    _log(f"dataset[idx={idx}] fetched in {dt:.3f}s", rank)
                self._seen += 1
                return item

        dataset = _DebugDataset(dataset, dbg_rank=rank, log_every=0, max_items=8)
        _log("Wrapped dataset with debug timings (first 8 items).", rank)
    _log(f"Dataset ready in {time.time() - t0:.2f}s", rank)

    # Policy + processors
    # Re-disable device_map='auto' at load time without changing library code.
    try:
        from transformers import AutoModelForImageTextToText as _AMFITT  # type: ignore

        _orig_from_pretrained = _AMFITT.from_pretrained

        def _patched_from_pretrained(*args, **kwargs):  # noqa: ANN001, ANN003
            # Force a single-device load (no model-parallel / offload) for distributed training
            kwargs["device_map"] = None
            # Transformers may implicitly set device_map when low_cpu_mem_usage=True; disable it
            kwargs["low_cpu_mem_usage"] = False
            return _orig_from_pretrained(*args, **kwargs)

        _AMFITT.from_pretrained = _patched_from_pretrained  # type: ignore
        if accelerator.is_main_process:
            _log("[accelerate] Patched Transformers to disable device_map='auto' for distributed training.")
    except Exception as _e:  # noqa: BLE001
        if accelerator.is_main_process:
            _log(f"[accelerate] Warning: could not patch Transformers device_map: {_e}")

    # Ensure per-process device is explicit for both policy and processors.
    cfg.policy.device = str(accelerator.device)

    _log("Instantiating policy ...", rank)
    t0 = time.time()
    policy = make_policy(cfg=cfg.policy, ds_meta=ds_meta)
    # Attach dataset stats to policy for pixel-error monitoring.
    if ds_meta is not None:
        setattr(policy, "_dataset_stats", ds_meta.stats)
    _log(f"Policy ready in {time.time() - t0:.2f}s", rank)
    # Match original train.py semantics for processor creation/overrides
    processor_kwargs = {}
    if not (getattr(cfg, "resume", False) and cfg.policy.pretrained_path):
        processor_kwargs["dataset_stats"] = ds_meta.stats

    pretrained_path = cfg.policy.pretrained_path
    if getattr(cfg.policy, "type", None) == "smolvla":
        pretrained_path = None
    elif cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": str(accelerator.device)},
            "normalizer_processor": {"stats": ds_meta.stats},
        }
        processor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {"stats": ds_meta.stats},
        }
    _log("Building processors ...", rank)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=pretrained_path,
        **processor_kwargs,
    )
    _log("Processors ready", rank)

    # Optimizer & scheduler from policy presets or CLI overrides
    _log("Building optimizer & scheduler ...", rank)
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    _log("Optimizer & scheduler ready", rank)

    # Resume training state (step, optimizer, scheduler) if requested
    start_step = 0
    if getattr(cfg, "resume", False):
        if getattr(cfg, "checkpoint_path", None) is None:
            _log("[resume] Missing cfg.checkpoint_path; cannot resume. Proceeding from step 0.", rank)
        else:
            try:
                start_step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)
                if accelerator.is_main_process:
                    _log(f"[resume] Loaded training state from {cfg.checkpoint_path} (start_step={start_step})")
            except Exception as e:  # noqa: BLE001
                _log(f"[resume] Failed to load training state: {e}. Proceeding from step 0.", rank)

    # Dataloader (Accelerate handles distributed sharding via prepare())
    _log("Building dataloader ...", rank)
    dataloader = _make_dataloader(dataset, cfg.batch_size, cfg.num_workers, accelerator)
    _log(f"Dataloader ready (num_workers={cfg.num_workers})", rank)

    # Prepare with Accelerate (moves to device, wraps DDP where relevant)
    _log("Calling accelerator.prepare(...) ...", rank)
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    _log("accelerator.prepare done", rank)

    # Training loop
    steps = cfg.steps
    save_ckpt = cfg.save_checkpoint
    save_freq = cfg.save_freq

    # Simple iterator
    data_iter = iter(dataloader)

    policy.train()
    step_t0 = time.time()
    for step in range(start_step + 1, steps + 1):
        try:
            if DEBUG_FETCH:
                _log("Fetching next batch ...", rank)
            batch = next(data_iter)
            if DEBUG_FETCH:
                _log("Batch fetched", rank)
        except StopIteration:
            # Recreate iterator per epoch; update sampler epoch for proper shuffling
            data_iter = iter(dataloader)
            if DEBUG_FETCH:
                _log("Epoch iterator reset; fetching batch ...", rank)
            batch = next(data_iter)
            if DEBUG_FETCH:
                _log("Batch fetched after reset", rank)

        # Apply policy preprocessor pipeline (tokenization, normalization, device move)
        batch = preprocessor(batch)

        # ===== Online Reflow Logic =====
        # Check if reflow is enabled and we're past the start step
        reflow_noise = None
        reflow_actions = None
        unwrapped_policy = accelerator.unwrap_model(policy)
        policy_cfg = unwrapped_policy.config

        reflow_enabled = getattr(policy_cfg, "reflow_enabled", False)
        reflow_prob = getattr(policy_cfg, "reflow_prob", 0.5)
        reflow_start_step = getattr(policy_cfg, "reflow_start_step", 0)

        use_reflow_this_step = (
            reflow_enabled
            and step >= reflow_start_step
            and random.random() < reflow_prob
        )

        if use_reflow_this_step:
            # Generate (noise, action) pairs using current model
            # Need to prepare inputs for generate_reflow_targets
            with torch.no_grad():
                images, img_masks = unwrapped_policy.prepare_images(batch)
                state = unwrapped_policy.prepare_state(batch)
                from lerobot.policies.smolvla.modeling_smolvla import (
                    OBS_LANGUAGE_TOKENS,
                    OBS_LANGUAGE_ATTENTION_MASK,
                )
                lang_tokens = batch[OBS_LANGUAGE_TOKENS]
                lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

                # Generate reflow targets (noise, generated_actions)
                reflow_noise, reflow_actions = unwrapped_policy.model.generate_reflow_targets(
                    images, img_masks, lang_tokens, lang_masks, state
                )

        # Mixed precision autocast context (must be created per-iteration; it's single-use)
        with accelerator.autocast():
            loss, output_dict = policy.forward(
                batch,
                reflow_noise=reflow_noise,
                reflow_actions=reflow_actions,
            )

        # Backward + step
        accelerator.backward(loss)
        # Clip before step
        # Use optimizer config's grad_clip_norm to align with original train.py
        torch.nn.utils.clip_grad_norm_(policy.parameters(), getattr(cfg.optimizer, "grad_clip_norm", 10.0))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if lr_scheduler is not None:
            # IMPORTANT: pass the global training step to avoid multi-process
            # over-stepping the scheduler (each process would otherwise call
            # step() and advance the internal epoch). By providing `step`, all
            # processes set the same epoch value, keeping LR aligned with the
            # intended schedule.
            lr_scheduler.step(step)

        # Periodic checkpoint on main process only
        if accelerator.is_main_process and save_ckpt and (step % save_freq == 0 or step == steps):
            # Build checkpoint dir for this step
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            # Unwrap the model from Accelerate/DDP for saving
            to_save = accelerator.unwrap_model(policy)
            # Save full checkpoint: cfg, policy, optimizer, scheduler, and processors
            save_checkpoint(
                checkpoint_dir,
                step,
                cfg,
                to_save,
                optimizer,
                lr_scheduler,
                preprocessor,
                postprocessor,
            )
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger is not None:
                wandb_logger.log_policy(checkpoint_dir)

        # Minimal logging
        if accelerator.is_main_process and step % max(1, getattr(cfg, "log_freq", 200)) == 0:
            dt = time.time() - step_t0
            # Core metrics for grounding tasks
            loss_fm = float(output_dict.get("loss_fm", 0.0))
            loss_action_recon = float(output_dict.get("loss_action_recon", 0.0))
            mse = float(output_dict.get("loss_mse", 0.0))
            l1 = float(output_dict.get("loss_l1", 0.0))
            pixel_err_x = float(output_dict.get("pixel_err_x", 0.0))
            pixel_err_y = float(output_dict.get("pixel_err_y", 0.0))
            pixel_err_avg = float(output_dict.get("pixel_err_avg", 0.0))
            is_reflow = float(output_dict.get("is_reflow", 0.0))

            # Simplified console output (only core metrics)
            reflow_tag = " [REFLOW]" if is_reflow > 0.5 else ""
            _log(
                f"[step {step}/{steps}]{reflow_tag} "
                f"loss={loss.item():.4f} fm={loss_fm:.4f} act_recon={loss_action_recon:.4f} "
                f"mse={mse:.4f} l1={l1:.4f} "
                f"px_err={pixel_err_avg:.1f}px (x={pixel_err_x:.1f} y={pixel_err_y:.1f}) "
                f"step_time={dt:.3f}s"
            )
            step_t0 = time.time()
            if wandb_logger is not None:
                wandb_logger.log_dict(
                    {
                        # Core losses
                        "loss": loss.item(),
                        "loss_fm": loss_fm,
                        "loss_action_recon": loss_action_recon,
                        "loss_mse": mse,
                        "loss_l1": l1,
                        # Pixel errors (grounding-specific)
                        "pixel_err_x": pixel_err_x,
                        "pixel_err_y": pixel_err_y,
                        "pixel_err_avg": pixel_err_avg,
                        # Training info
                        "lr": optimizer.param_groups[0]["lr"],
                        # Reflow info
                        "is_reflow": is_reflow,
                    },
                    step,
                )

    # Graceful shutdown
    try:
        accelerator.wait_for_everyone()
    except Exception:
        pass
    # Explicitly drop references to help DataLoader cleanly shutdown
    try:
        del data_iter
    except Exception:
        pass
    try:
        del dataloader
    except Exception:
        pass
    # Encourage prompt cleanup
    try:
        gc.collect()
    except Exception:
        pass
    # Try to clean up distributed process group to silence NCCL warning
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            _log("Destroying process group ...", rank)
            with contextlib.suppress(Exception):
                dist.barrier()
            with contextlib.suppress(Exception):
                dist.destroy_process_group()
    except Exception:
        pass

    if hasattr(accelerator, "end_training"):
        with contextlib.suppress(Exception):
            accelerator.end_training()

    if accelerator.is_main_process:
        _log("Training complete.")


def main():
    train()


if __name__ == "__main__":
    main()
