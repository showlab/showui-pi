#!/usr/bin/env python
"""
ScreenSpot-Pro evaluation: predict click point, check if it falls inside GT bbox.

Usage:
    PYTHONPATH=lerobot/src \
    python scripts/eval_screenspot_pro.py \
      --ckpt <path/to/pretrained_model> \
      --output_dir outputs/eval_screenspot_pro \
      --annotations_root <path/to/ScreenSpot-Pro/annotations> \
      --images_root <path/to/ScreenSpot-Pro/images>
"""

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors, make_policy_config


# ---------------------------------------------------------------------------
# ScreenSpot-Pro dataset (inlined here so it lives only in the eval context).
# ---------------------------------------------------------------------------

def _to_px(x_norm: float, y_norm: float) -> tuple[int, int]:
    x = int(round(x_norm * 1920.0))
    y = int(round(y_norm * 1080.0))
    x = 0 if x < 0 else (1919 if x > 1919 else x)
    y = 0 if y < 0 else (1079 if y > 1079 else y)
    return x, y


def _normalize_bbox(bbox: list[float], img_w: float, img_h: float) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError(f"bbox length must be 4, got {len(bbox)}")
    x0, y0, x1, y1 = bbox
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"invalid img_size: ({img_w}, {img_h})")
    return x0 / img_w, y0 / img_h, x1 / img_w, y1 / img_h


@dataclass
class _SSPSample:
    image_path: Path
    bbox_norm: tuple[float, float, float, float]
    task: str


class _SSPDataset(Dataset):
    """
    ScreenSpot-Pro evaluation dataset.

    Conventions:
    - annotations_root: directory containing *.json annotation files (one per
      application group). Each file is a list[dict] with keys:
        {"img_filename": "common_windows/screenshot_xxx.png",
         "bbox": [x0, y0, x1, y1],
         "instruction": "empty recycle bin",
         "img_size": [W, H], ...}
    - images_root: directory containing the images referenced by img_filename.
    - Returns observation.images.screen / observation.state / action /
      action_is_pad fields compatible with the policy preprocessor.
    """

    def __init__(self, annotations_root: Path, images_root: Path, chunk_size: int = 21):
        self.annotations_root = Path(annotations_root)
        self.images_root = Path(images_root)
        self.chunk_size = int(chunk_size)

        if not self.annotations_root.is_dir():
            raise ValueError(f"annotations_root is not a directory: {self.annotations_root}")
        if not self.images_root.is_dir():
            raise ValueError(f"images_root is not a directory: {self.images_root}")

        samples: list[_SSPSample] = []
        for json_path in sorted(self.annotations_root.glob("*.json")):
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                continue
            for rec in data:
                img_filename = rec.get("img_filename")
                bbox = rec.get("bbox")
                instr = rec.get("instruction")
                img_size = rec.get("img_size")
                if not isinstance(img_filename, str):
                    continue
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    continue
                if not (isinstance(img_size, list) and len(img_size) == 2):
                    continue
                if not isinstance(instr, str):
                    continue
                img_path = self.images_root / img_filename
                if not img_path.is_file():
                    continue
                try:
                    w = float(img_size[0])
                    h = float(img_size[1])
                    x0n, y0n, x1n, y1n = _normalize_bbox(bbox, w, h)
                except Exception:
                    continue
                samples.append(
                    _SSPSample(
                        image_path=img_path,
                        bbox_norm=(x0n, y0n, x1n, y1n),
                        task=instr,
                    )
                )
        if len(samples) == 0:
            raise RuntimeError(f"No samples parsed from {self.annotations_root}")
        self._samples: list[_SSPSample] = samples

    def __len__(self) -> int:
        return len(self._samples)

    def _load_image_1080p(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((1920, 1080), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8).copy()
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return t

    def __getitem__(self, idx: int) -> dict:
        s = self._samples[idx]
        img_t = self._load_image_1080p(s.image_path).unsqueeze(0)
        x0, y0, x1, y1 = s.bbox_norm

        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        x_px, y_px = _to_px(cx, cy)

        T = self.chunk_size
        state = torch.zeros((1, 3), dtype=torch.float32)
        action = torch.zeros((T, 3), dtype=torch.float32)
        is_pad = torch.ones(T, dtype=torch.bool)
        state[0] = torch.tensor([0.0, -1.0, -1.0], dtype=torch.float32)
        action[0] = torch.tensor([1.0, float(x_px), float(y_px)], dtype=torch.float32)
        is_pad[0] = False
        if T > 1:
            action[1] = torch.tensor([0.0, float(x_px), float(y_px)], dtype=torch.float32)
            is_pad[1] = False
        if T > 2:
            action[2:] = action[1].unsqueeze(0).expand(T - 2, -1)

        return {
            "observation.images.screen": img_t,
            "observation.state": state,
            "action": action,
            "action_is_pad": is_pad,
            "task": s.task,
            "observation.images.screen_is_pad": torch.zeros(1, dtype=torch.bool),
            "observation.state_is_pad": torch.zeros(1, dtype=torch.bool),
            "episode_index": torch.tensor(idx, dtype=torch.int64),
            "bbox_norm": torch.tensor([x0, y0, x1, y1], dtype=torch.float32),
            "frame_index": torch.tensor(0, dtype=torch.int64),
            "index": torch.tensor(0, dtype=torch.int64),
            "timestamp": torch.tensor(0.0, dtype=torch.float32),
            "task_index": torch.tensor(0, dtype=torch.int64),
        }


def load_policy(ckpt_dir: str, device: str = "cuda"):
    """Load policy from checkpoint directory."""
    ckpt_path = Path(ckpt_dir)

    # Load config
    with open(ckpt_path / "config.json") as f:
        raw_cfg = json.load(f)

    # Extract and remove 'type' from raw_cfg (it's the discriminator, not a config field)
    ptype = raw_cfg.pop("type", None)
    if not ptype:
        train_cfg_path = ckpt_path / "train_config.json"
        if train_cfg_path.exists():
            with open(train_cfg_path) as f:
                train_cfg = json.load(f)
            ptype = (train_cfg.get("policy") or {}).get("type")

    policy_cfg = make_policy_config(ptype, **raw_cfg)
    policy_cfg.pretrained_path = str(ckpt_path)
    policy_cfg.device = device

    # Load dataset metadata (needed by make_policy for feature shapes)
    ds_root = Path(os.environ.get("DEX_DS_ROOT", "data/dex"))
    ds_meta = LeRobotDatasetMetadata("dex", root=ds_root, revision="v3.0")

    # Build policy
    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy.eval()

    # Build processors from saved JSON files.
    # Only override device; normalizer/unnormalizer configs are already correct in the JSON.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(ckpt_path),
        preprocessor_overrides={
            "device_processor": {"device": device},
        },
    )

    return policy, preprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to pretrained_model directory")
    parser.add_argument("--output_dir", default="outputs/eval_screenspot_pro")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--viz_n", type=int, default=50, help="Number of samples to visualize")
    parser.add_argument("--annotations_root", required=True,
                        help="Path to ScreenSpot-Pro annotations directory")
    parser.add_argument("--images_root", required=True,
                        help="Path to ScreenSpot-Pro images directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    viz_dir = Path(args.output_dir) / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    print("Loading policy...", flush=True)
    policy, preprocessor = load_policy(args.ckpt, args.device)

    print("Loading ScreenSpot-Pro dataset...", flush=True)
    dataset = _SSPDataset(
        annotations_root=Path(args.annotations_root),
        images_root=Path(args.images_root),
        chunk_size=21,
    )
    print(f"  {len(dataset)} samples", flush=True)

    # Load raw annotations for per-app/group metrics
    raw_samples = []
    ann_root = Path(args.annotations_root)
    for json_path in sorted(ann_root.glob("*.json")):
        with json_path.open("r") as f:
            raw_samples.extend(json.load(f))

    # Metrics
    correct = 0
    total = 0
    per_app = defaultdict(lambda: {"correct": 0, "total": 0})
    per_group = defaultdict(lambda: {"correct": 0, "total": 0})
    per_ui_type = defaultdict(lambda: {"correct": 0, "total": 0})
    results = []

    print("Running evaluation...", flush=True)
    t0 = time.time()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for idx in range(len(dataset)):
            # Get sample directly (no DataLoader) so preprocessor can add batch dim
            sample = dataset[idx]

            raw = raw_samples[idx] if idx < len(raw_samples) else {}
            app = raw.get("application", "unknown")
            group = raw.get("group", "unknown")
            ui_type = raw.get("ui_type", "unknown")
            img_size = raw.get("img_size", [1920, 1080])
            gt_bbox = raw.get("bbox", [0, 0, 0, 0])  # [x0, y0, x1, y1] in original pixel coords

            # Preprocess (adds batch dim, tokenizes, moves to device, normalizes actions)
            batch = preprocessor(sample)

            # Predict action chunk
            pred_chunk = policy.predict_action_chunk(batch)  # (B, T, action_dim)
            if pred_chunk.ndim == 2:
                pred_chunk = pred_chunk.unsqueeze(0)

            # First action step: [btn, x_norm, y_norm] in [0,1] normalized space
            # (model trained with smolvla_fixed_xy_normalizer: x/1920, y/1080)
            pred_x_norm = pred_chunk[0, 0, 1].item()
            pred_y_norm = pred_chunk[0, 0, 2].item()

            # Fixed XY unnormalization back to 1920x1080 pixel space
            pred_x_1920 = pred_x_norm * 1920.0
            pred_y_1080 = pred_y_norm * 1080.0

            # Scale to original image size
            W_orig, H_orig = float(img_size[0]), float(img_size[1])
            pred_x = pred_x_1920 * (W_orig / 1920.0)
            pred_y = pred_y_1080 * (H_orig / 1080.0)

            # Check if prediction falls inside GT bbox
            x0, y0, x1, y1 = gt_bbox
            hit = (x0 <= pred_x <= x1) and (y0 <= pred_y <= y1)

            correct += int(hit)
            total += 1
            per_app[app]["correct"] += int(hit)
            per_app[app]["total"] += 1
            per_group[group]["correct"] += int(hit)
            per_group[group]["total"] += 1
            per_ui_type[ui_type]["correct"] += int(hit)
            per_ui_type[ui_type]["total"] += 1

            results.append({
                "id": raw.get("id", idx),
                "app": app,
                "group": group,
                "ui_type": ui_type,
                "instruction": raw.get("instruction", ""),
                "gt_bbox": gt_bbox,
                "pred_x": round(pred_x, 1),
                "pred_y": round(pred_y, 1),
                "hit": hit,
            })

            # Visualize some samples
            if idx < args.viz_n:
                img_path = Path(args.images_root) / raw.get("img_filename", "")
                if img_path.is_file():
                    img = Image.open(img_path).convert("RGB")
                    draw = ImageDraw.Draw(img)
                    draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)
                    r = 8
                    draw.ellipse([pred_x - r, pred_y - r, pred_x + r, pred_y + r], fill=(255, 0, 0))
                    label = f"{'HIT' if hit else 'MISS'} | {raw.get('instruction', '')[:60]}"
                    draw.text((10, 10), label, fill=(255, 255, 0))
                    img.save(viz_dir / f"{idx:04d}_{app}_{'hit' if hit else 'miss'}.png")

            if (idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  [{idx+1}/{len(dataset)}] acc={correct}/{total} ({100*correct/total:.1f}%) elapsed={elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0

    # Summary
    acc = 100.0 * correct / max(1, total)
    print(f"\n=== ScreenSpot-Pro Results ===")
    print(f"Overall: {correct}/{total} = {acc:.1f}%")

    print(f"\nPer group:")
    for g in sorted(per_group):
        c, t = per_group[g]["correct"], per_group[g]["total"]
        print(f"  {g}: {c}/{t} = {100*c/t:.1f}%")

    print(f"\nPer UI type:")
    for u in sorted(per_ui_type):
        c, t = per_ui_type[u]["correct"], per_ui_type[u]["total"]
        print(f"  {u}: {c}/{t} = {100*c/t:.1f}%")

    print(f"\nPer application:")
    for a in sorted(per_app):
        c, t = per_app[a]["correct"], per_app[a]["total"]
        print(f"  {a}: {c}/{t} = {100*c/t:.1f}%")

    # Save results
    summary = {
        "overall_accuracy": acc,
        "correct": correct,
        "total": total,
        "elapsed_s": round(elapsed, 1),
        "per_group": {g: {"acc": round(100 * v["correct"] / max(1, v["total"]), 1), **v} for g, v in per_group.items()},
        "per_ui_type": {u: {"acc": round(100 * v["correct"] / max(1, v["total"]), 1), **v} for u, v in per_ui_type.items()},
        "per_app": {a: {"acc": round(100 * v["correct"] / max(1, v["total"]), 1), **v} for a, v in per_app.items()},
    }
    with open(Path(args.output_dir) / "screenspot_pro_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(Path(args.output_dir) / "screenspot_pro_details.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
