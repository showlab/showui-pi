#!/usr/bin/env python
"""
DEX benchmark evaluation.

The benchmark is hosted as a HuggingFace dataset and is downloaded on first
run (default: `h-siyuan/ScreenDrag`, 505 episodes spread evenly across five
domains: captcha, premiere, desktop, powerpoint, handwriting). Each episode
is a 21-step drag, and success is measured by whether the model's predicted
(x, y) lands within 20 px of the ground-truth endpoint at any step of the
chunk (`endpoint_accuracy_best`).

Usage:
    python scripts/eval_dex.py \\
        --ckpt outputs/showui_pi/checkpoints/015000/pretrained_model \\
        --output_dir outputs/eval_dex

Set `--domain <name>` to evaluate a single domain; omit for all five.
"""

import argparse
import io
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors


THRESHOLD_PX = 20.0
DOMAINS = ["captcha", "premiere", "desktop", "powerpoint", "handwriting"]


# --------------- data loading ---------------


def download_benchmark(dataset_id: str) -> Path:
    """Download the LeRobot v3.0 metadata + data parquet files of the benchmark.

    Uses per-file `hf_hub_download` rather than `snapshot_download` because the
    upstream dataset also ships tens of thousands of raw PNGs that we don't
    need; `snapshot_download`'s file-listing phase would be slow even with
    `allow_patterns`.

    Returns the local path to the v3.0 dataset root (the directory that will
    contain `meta/` and `data/`).
    """
    api = HfApi()
    all_files = api.list_repo_files(repo_id=dataset_id, repo_type="dataset")
    wanted = [
        p for p in all_files
        if p.startswith("test/parquet/meta/") or p.startswith("test/parquet/data/")
    ]
    if not wanted:
        raise FileNotFoundError(
            f"{dataset_id} does not contain a test/parquet/ tree"
        )
    print(f"Downloading {len(wanted)} files from {dataset_id} ...", flush=True)
    local_root: Path | None = None
    for i, rel in enumerate(wanted, 1):
        local = Path(
            hf_hub_download(repo_id=dataset_id, filename=rel, repo_type="dataset")
        )
        # The snapshot root (i.e. parent of `test/`) is shared across files.
        if local_root is None:
            parts = local.parts
            for j, part in enumerate(parts):
                if part == "test":
                    local_root = Path(*parts[:j])
                    break
        if i == 1 or i == len(wanted) or i % 5 == 0:
            print(f"  [{i}/{len(wanted)}] {rel}", flush=True)

    if local_root is None:
        raise RuntimeError("Could not determine snapshot root from downloads")

    root = local_root / "test" / "parquet"
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"Expected LeRobot v3.0 metadata at {root}/meta/info.json"
        )
    return root


def load_benchmark_data(ds_root: Path) -> tuple[pd.DataFrame, dict[int, str], dict[int, str]]:
    """Load all parquet data, the domain map, and the task-index lookup."""
    parquet_dir = ds_root / "data"
    dfs = [pd.read_parquet(p) for p in sorted(parquet_dir.glob("chunk-*/file-*.parquet"))]
    if not dfs:
        raise FileNotFoundError(f"No parquet files found under {parquet_dir}")
    df = pd.concat(dfs, ignore_index=True)

    with (ds_root / "meta" / "domain_map.json").open() as f:
        dm_raw = json.load(f)
    domain_map = {int(k): str(v) for k, v in dm_raw.items()}

    tasks_df = pd.read_parquet(ds_root / "meta" / "tasks.parquet")
    # `tasks.parquet` uses the task string as the row index and stores
    # `task_index` as the only column. Invert to {task_index: task_text}.
    tasks_by_id: dict[int, str] = {
        int(tasks_df.iloc[i]["task_index"]): str(tasks_df.index[i])
        for i in range(len(tasks_df))
    }
    return df, domain_map, tasks_by_id


# --------------- policy loading ---------------


def load_policy(ckpt_dir: str, ds_root: Path, device: str):
    """Load a SmolVLA policy from a checkpoint, using the benchmark metadata
    for feature shapes, and build the saved pre/post processors."""
    ckpt_path = Path(ckpt_dir)
    with (ckpt_path / "config.json").open() as f:
        raw_cfg = json.load(f)
    ptype = raw_cfg.pop("type", None)
    if not ptype:
        train_cfg_path = ckpt_path / "train_config.json"
        if train_cfg_path.is_file():
            with train_cfg_path.open() as f:
                tc = json.load(f)
            ptype = (tc.get("policy") or {}).get("type")
    if not ptype:
        raise ValueError(f"Could not determine policy type from {ckpt_path}")

    policy_cfg = make_policy_config(ptype, **raw_cfg)
    policy_cfg.pretrained_path = str(ckpt_path)
    policy_cfg.device = device

    ds_meta = LeRobotDatasetMetadata("ScreenDrag", root=ds_root, revision="v3.0")

    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(ckpt_path),
        preprocessor_overrides={
            "device_processor": {"device": device},
        },
    )
    return policy, preprocessor, postprocessor


# --------------- inference ---------------


def _decode_image(img_cell) -> torch.Tensor:
    """Parquet image cell -> (1, 3, H, W) float32 in [0, 1]. HF serializes the
    image as a dict with a `bytes` key holding the encoded PNG."""
    if isinstance(img_cell, dict):
        raw = img_cell.get("bytes")
    else:
        raw = img_cell
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def run_episode(
    policy,
    preprocessor,
    postprocessor,
    episode_df: pd.DataFrame,
    tasks_by_id: dict[int, str],
) -> dict:
    """Closed-loop inference over a single 21-step episode.

    Ground truth endpoint: the last frame's `action[1:3]` (x, y in pixel
    coordinates). We measure the minimum xy distance between predicted and
    target across the full chunk (`endpoint_accuracy_best`) and the final-step
    distance (`endpoint_accuracy`).
    """
    ep = episode_df.sort_values("frame_index").reset_index(drop=True)
    last_action = np.asarray(ep.iloc[-1]["action"], dtype=np.float64)
    tgt_x = float(last_action[1])
    tgt_y = float(last_action[2])

    task_index = int(ep.iloc[0]["task_index"])
    task_text = tasks_by_id.get(task_index, "")

    prev_state = torch.tensor([[0.0, -1.0, -1.0]], dtype=torch.float32)
    best_err = float("inf")
    final_err = float("inf")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(len(ep)):
            row = ep.iloc[i]
            img = _decode_image(row["observation.images.screen"])

            batch = {
                "observation.images.screen": img,
                "observation.state": prev_state,
                "task": task_text,
            }
            mb = preprocessor(batch)
            act = policy.select_action(mb)
            act = postprocessor(act)
            a = act.numpy()[0]

            x_pred = float(np.clip(a[1], 0, 1919))
            y_pred = float(np.clip(a[2], 0, 1079))

            err = math.sqrt((x_pred - tgt_x) ** 2 + (y_pred - tgt_y) ** 2)
            if err < best_err:
                best_err = err
            final_err = err

            btn_pred = 1.0 if float(a[0]) > 0.5 else 0.0
            prev_state = torch.tensor(
                [[btn_pred, x_pred, y_pred]], dtype=torch.float32
            )

            # Force replanning at every step so state feedback actually matters.
            policy._queues["action"].clear()

    return {
        "endpoint_error_px_final": round(final_err, 2),
        "endpoint_error_px_best": round(best_err, 2),
        "endpoint_success": final_err <= THRESHOLD_PX,
        "endpoint_success_best": best_err <= THRESHOLD_PX,
    }


# --------------- main ---------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="Path to a pretrained_model directory (from a training checkpoint)")
    ap.add_argument("--output_dir", default="outputs/eval_dex")
    ap.add_argument("--domain", default=None,
                    choices=DOMAINS,
                    help="Evaluate a single domain; omit to evaluate all five")
    ap.add_argument("--hf_dataset", default="h-siyuan/ScreenDrag",
                    help="HuggingFace dataset ID for the benchmark")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ds_root = download_benchmark(args.hf_dataset)
    df, domain_map, tasks_by_id = load_benchmark_data(ds_root)
    print(f"Loaded {len(df)} frames across {df['episode_index'].nunique()} episodes",
          flush=True)

    print("Loading policy ...", flush=True)
    policy, preprocessor, postprocessor = load_policy(args.ckpt, ds_root, args.device)

    domains = [args.domain] if args.domain else DOMAINS
    all_results: dict[str, dict] = {}

    for domain in domains:
        domain_eps = sorted(ep for ep, d in domain_map.items() if d == domain)
        print(f"\n=== {domain}: {len(domain_eps)} episodes ===", flush=True)

        success = 0
        best_success = 0
        total = 0
        per_episode = []
        t0 = time.time()

        for ep_idx in domain_eps:
            episode_df = df[df["episode_index"] == ep_idx]
            if len(episode_df) == 0:
                print(f"  [skip] episode {ep_idx}: no frames in parquet", flush=True)
                continue
            r = run_episode(policy, preprocessor, postprocessor, episode_df, tasks_by_id)
            success += int(r["endpoint_success"])
            best_success += int(r["endpoint_success_best"])
            total += 1
            per_episode.append({"episode_index": int(ep_idx), **r})

            if total % 10 == 0 or total == len(domain_eps):
                elapsed = time.time() - t0
                sr = 100 * best_success / total
                print(f"  [{domain}] {total}/{len(domain_eps)}  sr_best={sr:.2f}%  elapsed={elapsed:.0f}s",
                      flush=True)

        summary = {
            "domain": domain,
            "mapping_size": len(domain_eps),
            "evaluated": total,
            "success": success,
            "success_best": best_success,
            "endpoint_accuracy": f"{100 * success / total:.2f}%" if total else "N/A",
            "endpoint_accuracy_best": f"{100 * best_success / total:.2f}%" if total else "N/A",
        }
        all_results[domain] = summary

        out = Path(args.output_dir) / domain
        out.mkdir(parents=True, exist_ok=True)
        with (out / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        with (out / "per_episode.json").open("w") as f:
            json.dump(per_episode, f, indent=2)

    print("\n=== Summary ===", flush=True)
    sum_best = 0.0
    n = 0
    for domain, r in all_results.items():
        acc = r["endpoint_accuracy_best"]
        print(f"  {domain}: {acc}  ({r['success_best']}/{r['evaluated']}, mapping={r['mapping_size']})")
        if acc != "N/A":
            sum_best += float(acc.rstrip("%"))
            n += 1
    if n > 1:
        avg = sum_best / n
        print(f"  DEX avg: {avg:.2f}%")
        with (Path(args.output_dir) / "aggregate.json").open("w") as f:
            json.dump(
                {"dex_avg_best": f"{avg:.2f}%", "per_domain": all_results},
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
