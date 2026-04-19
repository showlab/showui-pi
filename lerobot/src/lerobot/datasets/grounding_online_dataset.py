#!/usr/bin/env python3
"""
Online grounding datasets (single-click supervision).

Each loader reads a raw JSON annotation file plus a directory of images and
emits samples compatible with the LeRobotDataset __getitem__ contract:
  - observation.images.screen: (3, H, W) float32 in [0, 1]
  - observation.state: (T=21, 3)
  - action: (T=21, 3)
  - action_is_pad: (T,) bool
  - plus episode_index / frame_index / index / timestamp / task

Images are resized to 1920x1080. Click coordinates are read as normalized
(x_norm, y_norm) pairs and converted to pixel space via _to_px.
"""

from __future__ import annotations

import base64
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# Matches pyautogui.click(x=..., y=...) statements.
CLICK_RE = re.compile(r"pyautogui\.click\(x=([0-9]*\.?[0-9]+),\s*y=([0-9]*\.?[0-9]+)\)")


@dataclass
class UgroundClickSample:
    image: str
    x_norm: float
    y_norm: float
    bbox_norm: tuple[float, float, float, float]
    task: str


def _parse_uground_clicks(items: list[dict]) -> list[UgroundClickSample]:
    samples: list[UgroundClickSample] = []
    for obj in items:
        img = obj.get("image")
        conv = obj.get("conversations", [])
        if not img or not isinstance(conv, list):
            continue
        last_human: str | None = None
        for msg in conv:
            frm = msg.get("from")
            val = msg.get("value")
            if not isinstance(val, str):
                continue
            if frm == "human":
                last_human = val
                continue
            if frm != "gpt":
                continue
            m = CLICK_RE.search(val)
            if m is None:
                continue
            bbox = msg.get("bbox_gt")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                raise ValueError("sample is missing bbox_gt of length 4")
            try:
                x_norm = float(m.group(1))
                y_norm = float(m.group(2))
                x0 = float(bbox[0])
                y0 = float(bbox[1])
                x1 = float(bbox[2])
                y1 = float(bbox[3])
            except Exception:
                continue
            task = (last_human or "").lstrip()
            if task.startswith("<image>"):
                task = task[len("<image>") :].lstrip()
            samples.append(
                UgroundClickSample(
                    image=img,
                    x_norm=x_norm,
                    y_norm=y_norm,
                    bbox_norm=(x0, y0, x1, y1),
                    task=task,
                )
            )
    return samples


def _to_px(x_norm: float, y_norm: float) -> tuple[int, int]:
    # Map to 1920x1080 pixel space.
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


def _filter_portrait_samples(samples, path_of):  # noqa: ANN001
    cache: dict[str, bool] = {}
    kept = []
    for s in samples:
        p = path_of(s)
        k = str(p)
        if k not in cache:
            with Image.open(p) as im:
                w, h = im.size
            cache[k] = h > w
        if not cache[k]:
            kept.append(s)
    return kept


class UgroundOnlineDataset(Dataset):
    """
    Online UGround click dataset. Emits samples compatible with the LeRobot training contract.
    """

    def __init__(self, data_path: str, images_root: Path, chunk_size: int = 21, filter_portrait: bool = False):
        self.json_path = Path(data_path)
        self.images_root = Path(images_root)
        self.chunk_size = int(chunk_size)

        with self.json_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError(f"Expected list in {self.json_path}, got {type(raw)}")
        samples = _parse_uground_clicks(raw)
        if filter_portrait:
            root = self.images_root
            samples = _filter_portrait_samples(samples, lambda s: root / s.image)
        self._samples: List[UgroundClickSample] = samples

    def __len__(self) -> int:
        return len(self._samples)

    def _load_image_1080p(self, name: str) -> torch.Tensor:
        p = self.images_root / name
        img = Image.open(p).convert("RGB").resize((1920, 1080), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8).copy()
        # (H,W,3) -> (3,H,W), float [0,1]
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return t

    def __getitem__(self, idx: int) -> dict:
        s = self._samples[idx]
        img_t = self._load_image_1080p(s.image)  # (3,1080,1920)
        img_t = img_t.unsqueeze(0)  # (1,3,1080,1920)
        x_px, y_px = _to_px(s.x_norm, s.y_norm)
        x0, y0, x1, y1 = s.bbox_norm

        # Build state/action sequence: step 0 = mouse down, step 1 = mouse up, rest = padding.
        T = self.chunk_size
        state = torch.zeros((1, 3), dtype=torch.float32)
        action = torch.zeros((T, 3), dtype=torch.float32)
        is_pad = torch.ones(T, dtype=torch.bool)

        state[0] = torch.tensor([0.0, -1.0, -1.0], dtype=torch.float32)
        action[0] = torch.tensor([1.0, float(x_px), float(y_px)], dtype=torch.float32)  # down
        is_pad[0] = False
        if T > 1:
            action[1] = torch.tensor([0.0, float(x_px), float(y_px)], dtype=torch.float32)  # up
            is_pad[1] = False
        if T > 2:
            # Remaining steps are padding (not used in loss).
            action[2:] = action[1].unsqueeze(0).expand(T - 2, -1)

        item = {
            "observation.images.screen": img_t,  # (1,3,1080,1920)
            "observation.state": state,  # (1,3)
            "action": action,  # (T,3)
            "action_is_pad": is_pad,  # (T,)
            "task": s.task,
            # Padding masks.
            "observation.images.screen_is_pad": torch.zeros(1, dtype=torch.bool),
            "observation.state_is_pad": torch.zeros(1, dtype=torch.bool),
            # Compatibility fields.
            "episode_index": torch.tensor(idx, dtype=torch.int64),
            "bbox_norm": torch.tensor([x0, y0, x1, y1], dtype=torch.float32),
            "frame_index": torch.tensor(0, dtype=torch.int64),
            "index": torch.tensor(0, dtype=torch.int64),
            "timestamp": torch.tensor(0.0, dtype=torch.float32),
            "task_index": torch.tensor(0, dtype=torch.int64),
        }
        return item


@dataclass
class ClickSample:
    image: str
    x_norm: float
    y_norm: float
    bbox_norm: tuple[float, float, float, float]
    task: str


def _parse_clicks_generic(items: list[dict]) -> list[ClickSample]:
    samples: list[ClickSample] = []
    for obj in items:
        img = obj.get("image")
        conv = obj.get("conversations", [])
        if not img or not isinstance(conv, list):
            continue
        last_human: str | None = None
        for msg in conv:
            frm = msg.get("from")
            val = msg.get("value")
            if not isinstance(val, str):
                continue
            if frm == "human":
                last_human = val
                continue
            if frm != "gpt":
                continue
            m = CLICK_RE.search(val)
            if m is None:
                continue
            bbox = msg.get("bbox_gt")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                raise ValueError("sample is missing bbox_gt of length 4")
            try:
                x_norm = float(m.group(1))
                y_norm = float(m.group(2))
                x0 = float(bbox[0])
                y0 = float(bbox[1])
                x1 = float(bbox[2])
                y1 = float(bbox[3])
            except Exception:
                continue
            task = (last_human or "").lstrip()
            if task.startswith("<image>"):
                task = task[len("<image>") :].lstrip()
            samples.append(
                ClickSample(
                    image=img,
                    x_norm=x_norm,
                    y_norm=y_norm,
                    bbox_norm=(x0, y0, x1, y1),
                    task=task,
                )
            )
    return samples


class WaveUIOnlineDataset(Dataset):
    """Online Wave-UI click dataset."""

    def __init__(self, data_path: str, images_root: Path, chunk_size: int = 21, filter_portrait: bool = False):
        self.json_path = Path(data_path)
        self.images_root = Path(images_root)
        self.chunk_size = int(chunk_size)

        with self.json_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError(f"Expected list in {self.json_path}, got {type(raw)}")
        samples: List[ClickSample] = _parse_clicks_generic(raw)
        if filter_portrait:
            root = self.images_root
            samples = _filter_portrait_samples(samples, lambda s: root / s.image)
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def _load_image_1080p(self, name: str) -> torch.Tensor:
        p = self.images_root / name
        img = Image.open(p).convert("RGB").resize((1920, 1080), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8).copy()
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return t

    def __getitem__(self, idx: int) -> dict:
        s = self._samples[idx]
        img_t = self._load_image_1080p(s.image).unsqueeze(0)
        x_px, y_px = _to_px(s.x_norm, s.y_norm)
        x0, y0, x1, y1 = s.bbox_norm

        T = self.chunk_size
        state = torch.zeros((1, 3), dtype=torch.float32)
        action = torch.zeros((T, 3), dtype=torch.float32)
        is_pad = torch.ones(T, dtype=torch.bool)
        state[0] = torch.tensor([0.0, -1.0, -1.0], dtype=torch.float32)
        action[0] = torch.tensor([1.0, float(x_px), float(y_px)], dtype=torch.float32)  # down
        is_pad[0] = False
        if T > 1:
            action[1] = torch.tensor([0.0, float(x_px), float(y_px)], dtype=torch.float32)  # up
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

class GUIActOnlineDataset(Dataset):
    """Online GUIAct click dataset."""

    def __init__(self, data_path: str, images_root: Path, chunk_size: int = 21, filter_portrait: bool = False):
        self.annotations_dir = data_path
        self.images_root = Path(images_root)
        self.chunk_size = int(chunk_size)

        with Path(data_path).open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError(f"Expected list in {data_path}, got {type(raw)}")
        samples: List[ClickSample] = _parse_clicks_generic(raw)
        if filter_portrait:
            root = self.images_root
            samples = _filter_portrait_samples(samples, lambda s: root / s.image)
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def _load_image_1080p(self, name: str) -> torch.Tensor:
        p = self.images_root / name
        img = Image.open(p).convert("RGB").resize((1920, 1080), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8).copy()
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return t

    def __getitem__(self, idx: int) -> dict:
        s = self._samples[idx]
        img_t = self._load_image_1080p(s.image).unsqueeze(0)
        x_px, y_px = _to_px(s.x_norm, s.y_norm)
        x0, y0, x1, y1 = s.bbox_norm

        T = self.chunk_size
        state = torch.zeros((1, 3), dtype=torch.float32)
        action = torch.zeros((T, 3), dtype=torch.float32)
        is_pad = torch.ones(T, dtype=torch.bool)
        state[0] = torch.tensor([0.0, -1.0, -1.0], dtype=torch.float32)
        action[0] = torch.tensor([1.0, float(x_px), float(y_px)], dtype=torch.float32)  # down
        is_pad[0] = False
        if T > 1:
            action[1] = torch.tensor([0.0, float(x_px), float(y_px)], dtype=torch.float32)  # up
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


class ShowUIOnlineDataset(Dataset):
    """Online ShowUI-desktop click dataset."""

    def __init__(self, data_path: str, images_root: Path, chunk_size: int = 21, filter_portrait: bool = False):
        self.data_path = data_path
        self.images_root = Path(images_root)
        self.chunk_size = int(chunk_size)

        with Path(data_path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        samples: List[ClickSample] = []
        for rec in data:
            img_url = rec.get("img_url")
            elems = rec.get("element") or []
            if not isinstance(img_url, str):
                continue
            for e in elems:
                instr = e.get("instruction")
                pt = e.get("point")
                bbox = e.get("bbox")
                if not isinstance(instr, str):
                    continue
                if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                    continue
                if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                    raise ValueError("sample is missing bbox of length 4")
                x_norm, y_norm = float(pt[0]), float(pt[1])
                x0, y0, x1, y1 = (
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                )
                # Strip screenshots/ prefix from image path.
                rel = img_url
                if rel.startswith("screenshots/"):
                    rel = rel[len("screenshots/") :]
                samples.append(
                    ClickSample(
                        image=rel,
                        x_norm=x_norm,
                        y_norm=y_norm,
                        bbox_norm=(x0, y0, x1, y1),
                        task=instr,
                    )
                )
        if filter_portrait:
            root = self.images_root
            samples = _filter_portrait_samples(samples, lambda s: root / s.image)
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def _load_image_1080p(self, rel: str) -> torch.Tensor:
        p = self.images_root / rel
        img = Image.open(p).convert("RGB").resize((1920, 1080), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8).copy()
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return t

    def __getitem__(self, idx: int) -> dict:
        s = self._samples[idx]
        img_t = self._load_image_1080p(s.image).unsqueeze(0)
        x_px, y_px = _to_px(s.x_norm, s.y_norm)
        x0, y0, x1, y1 = s.bbox_norm

        T = self.chunk_size
        state = torch.zeros((1, 3), dtype=torch.float32)
        action = torch.zeros((T, 3), dtype=torch.float32)
        is_pad = torch.ones(T, dtype=torch.bool)
        state[0] = torch.tensor([0.0, -1.0, -1.0], dtype=torch.float32)
        action[0] = torch.tensor([1.0, float(x_px), float(y_px)], dtype=torch.float32)  # down
        is_pad[0] = False
        if T > 1:
            action[1] = torch.tensor([0.0, float(x_px), float(y_px)], dtype=torch.float32)  # up
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


class SEGUIOnlineDataset(Dataset):
    """Online SE-GUI click dataset."""

    def __init__(self, data_path: str, images_root: Path, chunk_size: int = 21, filter_portrait: bool = False):
        self.data_path = data_path
        self.images_root = Path(images_root)
        self.chunk_size = int(chunk_size)

        raw = json.loads(Path(data_path).read_text())
        if not isinstance(raw, list):
            raise ValueError(f"Expected list in {data_path}, got {type(raw)}")

        samples: List[ClickSample] = []
        for rec in raw:
            image = rec.get("image")
            x_norm = rec.get("x_norm")
            y_norm = rec.get("y_norm")
            bbox = rec.get("bbox_norm")
            task = rec.get("task")
            if not isinstance(image, str):
                continue
            if not isinstance(task, str):
                continue
            if not isinstance(x_norm, (float, int)) or not isinstance(y_norm, (float, int)):
                continue
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            samples.append(
                ClickSample(
                    image=image,
                    x_norm=float(x_norm),
                    y_norm=float(y_norm),
                    bbox_norm=(x0, y0, x1, y1),
                    task=task,
                )
            )

        if filter_portrait:
            root = self.images_root
            samples = _filter_portrait_samples(samples, lambda s: root / s.image)
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def _load_image_1080p(self, rel: str) -> torch.Tensor:
        p = self.images_root / rel
        img = Image.open(p).convert("RGB").resize((1920, 1080), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8).copy()
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return t

    def __getitem__(self, idx: int) -> dict:
        s = self._samples[idx]
        img_t = self._load_image_1080p(s.image).unsqueeze(0)
        x_px, y_px = _to_px(s.x_norm, s.y_norm)
        x0, y0, x1, y1 = s.bbox_norm

        T = self.chunk_size
        state = torch.zeros((1, 3), dtype=torch.float32)
        action = torch.zeros((T, 3), dtype=torch.float32)
        is_pad = torch.ones(T, dtype=torch.bool)

        state[0] = torch.tensor([0.0, -1.0, -1.0], dtype=torch.float32)
        action[0] = torch.tensor([1.0, float(x_px), float(y_px)], dtype=torch.float32)  # down
        is_pad[0] = False
        if T > 1:
            action[1] = torch.tensor([0.0, float(x_px), float(y_px)], dtype=torch.float32)  # up
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

