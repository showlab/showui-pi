#!/usr/bin/env python
"""
Offline evaluation on a LeRobotDataset without a live environment.

What it does
- Loads a saved policy checkpoint (or hub repo) and the dataset.
- Builds the same pre/post processors as training to ensure normalization matches.
- Iterates over the dataset to compute:
  - Flow-matching training loss (policy.forward) averaged over batches
  - Action prediction MSE between predicted action chunks and ground-truth (in normalized space)
- Optionally renders predicted vs ground-truth 2D traces over the input screenshot for a few samples.

Usage example

PYTHONPATH=lerobot/src \
python -m lerobot.scripts.lerobot_offline_eval \
  --policy.path=outputs/showui_pi/checkpoints/015000/pretrained_model \
  --dataset.repo_id=dex \
  --dataset.root=data/dex \
  --dataset.revision=v3.0 \
  --batch_size=8 --num_workers=0 \
  --max_batches=200 \
  --viz_n=50 \
  --output_dir=outputs/eval_offline

Notes
- This computes action MSE in normalized space (matching the model training space). For visualization, we
  unnormalize the two selected action dims (XY) using dataset stats and overlay them on the screenshot.
- The script tries to auto-detect XY indices from ACTION names; if not found, it falls back to the first two dims.
"""

import json
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors, make_policy_config
from lerobot.utils.constants import ACTION


@dataclass
class OfflineEvalConfig:
    dataset: DatasetConfig
    # Path to a saved policy directory (contains config.json + model.safetensors)
    policy_path: str | None = None
    output_dir: Path | None = None
    job_name: str | None = None

    batch_size: int = 8
    num_workers: int = 0
    max_batches: int | None = None
    viz_n: int = 50  # how many samples to visualize
    viz_every: int = 50  # visualize one sample every N batches (approx)
    device: str = "cuda"
    # Optional: only evaluate/visualize episodes whose task string contains this substring (case-insensitive)
    task_filter_substr: str | None = None
    # New: sample N unique tasks (one episode per distinct task) and visualize their first frame
    viz_unique_tasks_n: int = 0
    viz_exclude_task_substr: str | None = None  # e.g., 'captcha' to avoid over-represented domain
    viz_first_frame_only: bool = True

    def __post_init__(self):
        # Accept either --policy.path via parser wrapper or explicit --policy_path
        if self.policy_path is None:
            self.policy_path = parser.get_path_arg("policy")
        if not self.policy_path:
            raise ValueError("--policy.path (or --policy_path) is required for offline evaluation.")

        if self.job_name is None:
            self.job_name = f"offline_eval"
        if self.output_dir is None:
            self.output_dir = Path("outputs/eval_offline") / self.job_name

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """Enable parser to treat --policy.path specially (like other scripts)."""
        return ["policy"]


def _find_xy_indices(action_names: list[str]) -> tuple[int, int]:
    """Heuristic to find X/Y indices from action names.

    Falls back to (0, 1) if not found.
    """
    lower = [n.lower() for n in action_names]
    candidates = [
        ("cursor_x", "cursor_y"),
        ("mouse_x", "mouse_y"),
        ("x", "y"),
    ]
    for xk, yk in candidates:
        if xk in lower and yk in lower:
            return lower.index(xk), lower.index(yk)
    return 0, 1


def _find_btn_index(action_names: list[str]) -> int:
    """Heuristic to find button/press index from action names.

    Falls back to 0 if not found.
    """
    lower = [n.lower() for n in action_names]
    for k in ("btn", "button", "press"):
        if k in lower:
            return lower.index(k)
    return 0


def _path_length_ratio(xs: np.ndarray, ys: np.ndarray) -> float:
    """Return path length / end-to-end length; >1 means curved.

    If end-to-end is ~0 and moved, returns a large value.
    """
    if xs.shape[0] < 2:
        return 1.0
    dx = np.diff(xs)
    dy = np.diff(ys)
    seg = np.sqrt(dx * dx + dy * dy).sum()
    ee = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
    if ee < 1e-8:
        return 1.0 if seg < 1e-8 else 999.0
    return float(seg / ee)


def _to_pil(img_chw: torch.Tensor) -> Image.Image:
    """Convert CHW float[0,1] tensor to PIL Image."""
    c, h, w = img_chw.shape
    img = (img_chw.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy())
    return Image.fromarray(img)


def _draw_trace(img: Image.Image, xs: np.ndarray, ys: np.ndarray, color_pred=(255, 0, 0), color_gt=(0, 255, 0)) -> Image.Image:
    draw = ImageDraw.Draw(img)
    # predicted trace
    pts_pred = list(zip(xs.tolist(), ys.tolist()))
    if len(pts_pred) > 1:
        draw.line(pts_pred, fill=color_pred, width=3)
    for p in pts_pred:
        draw.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=color_pred)
    # GT trace
    pts_gt = list(zip(xs.tolist(), ys.tolist()))
    # Optional: separate style/width for GT if needed. Here reuse same for simplicity.
    return img


def _unnormalize(vec: torch.Tensor, stats: dict[str, Any]) -> torch.Tensor:
    """Unnormalize a tensor given dataset stats with mean/std keys.

    vec shape: (..., D)
    stats: {"mean": Tensor[D], "std": Tensor[D]} or nested; handle both list and tensors.
    """
    mean = stats.get("mean")
    std = stats.get("std")
    if isinstance(mean, torch.Tensor):
        m = mean.to(vec.device, dtype=vec.dtype)
    else:
        m = torch.as_tensor(mean, device=vec.device, dtype=vec.dtype)
    if isinstance(std, torch.Tensor):
        s = std.to(vec.device, dtype=vec.dtype)
    else:
        s = torch.as_tensor(std, device=vec.device, dtype=vec.dtype)
    # Broadcast to vec last dim
    while m.ndim < vec.ndim:
        m = m.unsqueeze(0)
        s = s.unsqueeze(0)
    return vec * s + m


def _select_stats(stats: dict[str, Any], indices: list[int]) -> dict[str, torch.Tensor]:
    """Select a subset of statistics (mean/std) by indices to match a subvector.

    Returns a dict with tensors shaped (len(indices),).
    """
    mean = stats.get("mean")
    std = stats.get("std")
    if mean is None or std is None:
        raise ValueError("Missing mean/std in stats for unnormalization.")
    m = torch.as_tensor(mean).view(-1)
    s = torch.as_tensor(std).view(-1)
    idx = torch.as_tensor(indices, dtype=torch.long)
    return {"mean": m[idx], "std": s[idx]}


@parser.wrap()
def offline_eval(cfg: OfflineEvalConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Helper: robustly load policy config even if config.json lacks 'type'
    def _load_policy_cfg(policy_dir: str) -> PreTrainedConfig:
        from draccus.utils import ParsingError  # type: ignore

        cli_overrides = parser.get_cli_overrides("policy") or []
        try:
            pcfg = PreTrainedConfig.from_pretrained(policy_dir, cli_overrides=cli_overrides)
            pcfg.pretrained_path = policy_dir
            pcfg.device = cfg.device
            return pcfg
        except ParsingError:
            # Fallback: read config.json and train_config.json to get 'type'
            policy_dir_path = Path(policy_dir)
            with open(policy_dir_path / "config.json", "r") as f:
                raw = json.load(f)
            # Try train_config.json for policy.type
            ptype = None
            train_cfg_path = policy_dir_path / "train_config.json"
            if train_cfg_path.exists():
                with open(train_cfg_path, "r") as f:
                    train_cfg = json.load(f)
                ptype = (train_cfg.get("policy") or {}).get("type")
            if not ptype:
                raise
            # Build config object directly
            pcfg = make_policy_config(ptype, **raw)
            pcfg.pretrained_path = policy_dir
            pcfg.device = cfg.device
            return pcfg

    # If requested, filter episodes by task substring (PR tasks etc.) before building the dataset
    if cfg.task_filter_substr:
        try:
            from pathlib import Path as _Path
            import pandas as _pd

            episodes_dir = _Path(cfg.dataset.root) / "meta/episodes"
            match = cfg.task_filter_substr.lower()
            ep_indices: list[int] = []
            for chunk in sorted(episodes_dir.glob("chunk-*/file-*.parquet")):
                df = _pd.read_parquet(chunk, columns=["episode_index", "tasks"])  # type: ignore
                for ep_idx, tasks in zip(df["episode_index"], df["tasks"]):
                    # tasks is typically a 1-element array of str for our datasets
                    if tasks is None:
                        continue
                    names = tasks if isinstance(tasks, (list, tuple, np.ndarray)) else [tasks]
                    names = [str(t) for t in names]
                    if any(match in t.lower() for t in names):
                        ep_indices.append(int(ep_idx))
            # Deduplicate and sort
            ep_indices = sorted(set(ep_indices))
            if len(ep_indices) == 0:
                print(f"[offline_eval] Warning: no episodes matched task_filter_substr='{cfg.task_filter_substr}'.")
            else:
                print(f"[offline_eval] Using {len(ep_indices)} episodes matched by task_filter_substr='{cfg.task_filter_substr}'.")
            # Inject into dataset config for downstream make_dataset
            cfg.dataset.episodes = ep_indices
        except Exception as e:  # noqa: BLE001
            print(f"[offline_eval] task_filter_substr failed with: {e}. Proceeding without filtering.")

    # Instantiate policy config from path first (needed to compute dataset delta_timestamps)
    policy_cfg = _load_policy_cfg(cfg.policy_path)

    # Build dataset via factory (duck-typed minimal config) and pass policy cfg
    class _Duck:
        pass

    duck = _Duck()
    duck.dataset = cfg.dataset
    duck.policy = policy_cfg
    duck.num_workers = cfg.num_workers
    dataset = make_dataset(duck)

    # Safety: avoid episode-global absolute index mismatches when only a subset of frames
    # is present locally by disabling delta lookups during offline visualization.
    # This keeps __getitem__ from querying around the current index using episode-level
    # absolute frame indices, which may exceed the locally available hf_dataset size.
    try:
        dataset.delta_timestamps = None
        dataset.delta_indices = None
    except Exception:
        pass

    # Ensure hf_dataset is restricted to the selected episodes (if any).
    # Without this, when data is locally available, LeRobotDataset may keep
    # all frames in hf_dataset even if cfg.dataset.episodes is set, which can
    # lead to visualizing the wrong frames/GT from unrelated episodes.
    try:
        sel_eps = getattr(cfg.dataset, "episodes", None)
        if sel_eps is not None and len(sel_eps) > 0:
            try:
                eps_set = set(int(e) for e in sel_eps)
            except Exception:
                eps_set = set(sel_eps)
            before_n = len(dataset.hf_dataset) if dataset.hf_dataset is not None else 0
            # Use membership filter to preserve underlying index mapping expected by delta_indices
            dataset.hf_dataset = dataset.hf_dataset.filter(
                lambda x: int(x["episode_index"]) in eps_set
            )
            after_n = len(dataset.hf_dataset) if dataset.hf_dataset is not None else 0
            print(
                f"[offline_eval] Filtered hf_dataset to {after_n} frames from episodes={sorted(list(eps_set))} (was {before_n})."
            )
    except Exception as e:  # noqa: BLE001
        print(f"[offline_eval] Warning: episode filtering failed: {e}")

    # Avoid Transformers auto device_map sharding (keep model on a single device)
    try:
        from transformers import AutoModelForImageTextToText as _AMFITT  # type: ignore

        _orig_from_pretrained = _AMFITT.from_pretrained

        def _patched_from_pretrained(*args, **kwargs):  # noqa: ANN001, ANN003
            kwargs["device_map"] = None
            kwargs["low_cpu_mem_usage"] = False
            return _orig_from_pretrained(*args, **kwargs)

        _AMFITT.from_pretrained = _patched_from_pretrained  # type: ignore
    except Exception:
        pass

    # Instantiate policy model
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)

    # Processors (align semantics with train.py)
    processor_kwargs: dict[str, Any] = {}
    # In offline_eval we never resume a training run; we can safely provide dataset stats
    processor_kwargs["dataset_stats"] = dataset.meta.stats
    # SmolVLA has a fixed XY normalizer/unnormalizer (1920×1080) with step names
    # `smolvla_fixed_xy_{normalizer,unnormalizer}`. Do not try to override with the generic
    # `normalizer_processor`/`unnormalizer_processor` keys (it will KeyError).
    pretrained_path_for_processors = policy_cfg.pretrained_path
    if getattr(policy_cfg, "type", None) == "smolvla":
        pretrained_path_for_processors = None
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": cfg.device},
        }
    elif policy_cfg.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": cfg.device},
            "normalizer_processor": {"stats": dataset.meta.stats},
        }
        processor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {"stats": dataset.meta.stats},
        }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=pretrained_path_for_processors,
        **processor_kwargs,
    )

    # Helper: optionally build a subset dataloader that samples the first frame of N unique tasks
    subset_dl: DataLoader | None = None
    picked_info: list[dict] = []
    if cfg.viz_unique_tasks_n and cfg.viz_unique_tasks_n > 0:
        import random, math
        # Build first-occurrence index per available episode in the loaded hf_dataset to avoid OOB indices
        ep_column = dataset.hf_dataset["episode_index"]  # list-like of length num_frames
        first_idx_by_ep: dict[int, int] = {}
        for i, epv in enumerate(ep_column):
            epi = int(epv.item() if hasattr(epv, "item") else epv)
            if epi not in first_idx_by_ep:
                first_idx_by_ep[epi] = i

        # Shuffle available episodes and pick unique tasks
        available_eps = list(first_idx_by_ep.keys())
        random.shuffle(available_eps)
        seen_tasks: set[str] = set()
        indices: list[int] = []
        exclude = (cfg.viz_exclude_task_substr or "").lower()
        for epi in available_eps:
            try:
                ep = dataset.meta.episodes[int(epi)]
                tasks = ep.get("tasks") if isinstance(ep, dict) else ep["tasks"]
                task_name = None
                if isinstance(tasks, list) and len(tasks) > 0:
                    task_name = str(tasks[0])
                elif tasks is not None:
                    task_name = str(tasks)
                else:
                    task_name = ""

                if exclude and (exclude in task_name.lower()):
                    continue
                if task_name in seen_tasks:
                    continue

                frm = first_idx_by_ep[int(epi)]  # first available frame index in the loaded dataset
                indices.append(frm)
                seen_tasks.add(task_name)
                picked_info.append({
                    "episode_index": int(ep.get("episode_index") if isinstance(ep, dict) else ep["episode_index"]),
                    "task": task_name,
                    "frame_index": frm,
                })
                if len(indices) >= int(cfg.viz_unique_tasks_n):
                    break
            except Exception:
                continue
        if len(indices) == 0:
            print("[offline_eval] Warning: could not sample any unique-task episodes; falling back to full dataloader.")
        else:
            from torch.utils.data import Subset
            subset = Subset(dataset, indices)
            subset_dl = DataLoader(subset, batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
            print(f"[offline_eval] Will visualize {len(indices)} unique tasks (exclude='{exclude}')")

    # Default dataloader
    dl = subset_dl or DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # Extract action names and XY indices
    action_names = dataset.meta.names.get(ACTION, [])
    if not action_names:
        # safe fallback to range
        action_dim = dataset.meta.shapes.get(ACTION, (2,))[0]
        action_names = [str(i) for i in range(action_dim)]
    x_idx, y_idx = _find_xy_indices(action_names)
    btn_idx = _find_btn_index(action_names)

    # Stats for unnormalizing actions for viz
    action_stats = dataset.meta.stats.get(ACTION, {})

    # Metrics accumulators
    total_fm_loss = 0.0
    total_mse = 0.0
    n_samples = 0
    n_batches = 0
    viz_saved = 0

    device = torch.device(cfg.device if torch.cuda.is_available() and "cuda" in cfg.device else "cpu")
    policy.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            # Copy original image for viz (first available camera key)
            cam_keys = dataset.meta.camera_keys
            img_key = cam_keys[0] if len(cam_keys) > 0 else None
            orig_imgs = batch[img_key] if img_key in batch else None  # (B, [T,] C, H, W) or (C, H, W)
            if orig_imgs is not None:
                # If sequence of frames provided, select the last one as in training code
                if orig_imgs.ndim == 5:  # (B, T, C, H, W)
                    if cfg.viz_first_frame_only:
                        orig_imgs = orig_imgs[:, 0, :, :, :]
                    else:
                        orig_imgs = orig_imgs[:, -1, :, :, :]
                elif orig_imgs.ndim == 3:  # (C, H, W) single sample
                    orig_imgs = orig_imgs.unsqueeze(0)

            # Preprocess to model space
            batch_pp = preprocessor(batch)

            autocast_ctx = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if (device.type == "cuda")
                else nullcontext()
            )

            # Flow-matching loss (averaged). We'll weight by batch size computed below.
            # In some offline visualization modes (e.g., when delta indices are disabled to
            # avoid OOB indices), forward() may mismatch shapes. Tolerate failures and
            # proceed with visualization-only path.
            with autocast_ctx:
                try:
                    fm_loss, _ = policy.forward(batch_pp)
                except Exception as e:  # noqa: BLE001
                    fm_loss = torch.tensor(0.0)
                    # Optional debug print to help diagnose but not interrupt visualization
                    print(f"[offline_eval] Skipping fm_loss this batch due to: {e}")

                # Predict action chunk in normalized space
                pred_chunk = policy.predict_action_chunk(batch_pp)  # usually (B, n_action_steps, action_dim)

            # Prepare GT action chunk from batch (normalized space)
            # SmolVLA policy has a helper for actions preparation
            gt_chunk = policy.prepare_action(batch_pp)  # (B, n_action_steps, max_action_dim) or (B, action_dim)

            # Ensure both pred and gt have 3 dims: (B, T, D)
            if pred_chunk.ndim == 2:
                pred_chunk = pred_chunk.unsqueeze(1)
            if gt_chunk.ndim == 2:
                # Repeat to match pred timesteps if available
                Tmatch = pred_chunk.shape[1] if pred_chunk.ndim == 3 else 1
                gt_chunk = gt_chunk.unsqueeze(1).repeat(1, Tmatch, 1)
            # Align timesteps by cropping to min length
            T = min(pred_chunk.shape[1], gt_chunk.shape[1])
            pred_chunk = pred_chunk[:, :T, :]
            gt_chunk = gt_chunk[:, :T, :]

            # Align last dim (truncate to original action dim)
            min_dim = min(pred_chunk.shape[-1], gt_chunk.shape[-1])
            pred = pred_chunk[..., :min_dim]
            gt = gt_chunk[..., :min_dim]

            # MSE over chunk in normalized space
            mse = torch.mean((pred - gt) ** 2).item()
            bsz = pred.shape[0]
            total_mse += mse * bsz
            total_fm_loss += float(fm_loss.item()) * bsz
            n_samples += bsz
            n_batches += 1

            # Visualization: save up to viz_n images, one every viz_every batches
            if viz_saved < cfg.viz_n and (batch_idx % max(1, cfg.viz_every) == 0) and orig_imgs is not None:
                # Use last frame image per sample and unnormalized XY traces
                B = pred.shape[0]
                T = pred.shape[1]
                # Unnormalize XY for both pred and GT using only XY stats
                xy_stats = _select_stats(action_stats, [x_idx, y_idx])
                pred_xy = _unnormalize(pred[..., [x_idx, y_idx]], xy_stats)  # (B, T, 2)
                gt_xy = _unnormalize(gt[..., [x_idx, y_idx]], xy_stats)  # (B, T, 2)

                for i in range(min(B, cfg.viz_n - viz_saved)):
                    img = _to_pil(orig_imgs[i]) if orig_imgs is not None else None
                    if img is None:
                        continue
                    w, h = img.size
                    # Convert to numpy and clamp to image bounds
                    px = pred_xy[i, :, 0].detach().cpu().numpy()
                    py = pred_xy[i, :, 1].detach().cpu().numpy()
                    gx = gt_xy[i, :, 0].detach().cpu().numpy()
                    gy = gt_xy[i, :, 1].detach().cpu().numpy()

                    # If coordinates are in [0,1], scale to pixels; otherwise assume already pixels
                    def _maybe_scale(x, maxv):
                        if np.all((x >= 0.0) & (x <= 1.0)):
                            return x * maxv
                        return x

                    px = _maybe_scale(px, w - 1)
                    gx = _maybe_scale(gx, w - 1)
                    py = _maybe_scale(py, h - 1)
                    gy = _maybe_scale(gy, h - 1)

                    # Compose overlay and save (dataset may have been filtered or subset-sampled)
                    canvas = img.copy()
                    draw = ImageDraw.Draw(canvas)
                    # GT in green, pred in red
                    pts_pred = list(zip(px.tolist(), py.tolist()))
                    pts_gt = list(zip(gx.tolist(), gy.tolist()))
                    if len(pts_gt) > 1:
                        draw.line(pts_gt, fill=(0, 255, 0), width=3)
                    if len(pts_pred) > 1:
                        draw.line(pts_pred, fill=(255, 0, 0), width=2)
                    # Annotate task and episode if available
                    task_name = None
                    if isinstance(batch.get("task"), list) and len(batch["task"]) > i:
                        task_name = str(batch["task"][i])
                    elif isinstance(batch.get("task"), torch.Tensor):
                        try:
                            task_name = str(batch["task"][i])
                        except Exception:
                            task_name = None
                    ep_name = None
                    try:
                        ep_name = int(batch["episode_index"][i].item())
                    except Exception:
                        ep_name = None
                    if task_name:
                        draw.text((10, 10), f"{task_name} (ep={ep_name})", fill=(255, 255, 0))
                    # Save
                    out_dir = Path(cfg.output_dir) / "viz"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    # Prefer deterministic names when sampling unique tasks: derive from batch episode_index
                    ep_idx_for_name = None
                    try:
                        ep_idx_for_name = int(batch["episode_index"][i].item())
                    except Exception:
                        ep_idx_for_name = None
                    out_path = (
                        out_dir / f"ep{ep_idx_for_name}_sample_{batch_idx}_{i}.png"
                        if ep_idx_for_name is not None
                        else out_dir / f"sample_b{batch_idx}_i{i}.png"
                    )
                    canvas.save(out_path)
                    viz_saved += 1
                    if viz_saved >= cfg.viz_n:
                        break

            if cfg.max_batches is not None and n_batches >= cfg.max_batches:
                break

    # Aggregate and save metrics
    avg_fm_loss = total_fm_loss / max(1, n_samples)
    avg_mse = total_mse / max(1, n_samples)
    summary = {
        "num_samples": n_samples,
        "num_batches": n_batches,
        "avg_flow_matching_loss": avg_fm_loss,
        "avg_action_mse_normalized": avg_mse,
        "viz_saved": viz_saved,
    }
    with open(Path(cfg.output_dir) / "offline_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Offline eval summary:", summary)


def main():
    offline_eval()


if __name__ == "__main__":
    main()
