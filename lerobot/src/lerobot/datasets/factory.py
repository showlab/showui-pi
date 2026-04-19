#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata, MultiLeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, OBS_PREFIX, REWARD

from .grounding_online_dataset import (
    UgroundOnlineDataset,
    WaveUIOnlineDataset,
    GUIActOnlineDataset,
    ShowUIOnlineDataset,
    SEGUIOnlineDataset,
)


class _GroundingConcatDataset(torch.utils.data.Dataset):  # type: ignore[misc]
    """Concatenates a base drag dataset with online grounding datasets."""

    def __init__(self, base, onlines: list[torch.utils.data.Dataset], meta):
        self._datasets: list[torch.utils.data.Dataset] = []
        if base is not None:
            self._datasets.append(base)
        self._datasets.extend(onlines)
        self.meta = meta

    def __len__(self) -> int:
        return sum(len(d) for d in self._datasets)

    def __getitem__(self, idx: int):
        for d in self._datasets:
            if idx < len(d):
                return d[idx]
            idx -= len(d)
        raise IndexError(idx)


class _ConcatDataset(torch.utils.data.Dataset):  # type: ignore[misc]
    """Simple concatenation of multiple torch datasets into a single index space."""

    def __init__(self, sources: list[torch.utils.data.Dataset], meta):
        self.meta = meta
        self._datasets = sources
        self._offsets = [0]
        for d in sources:
            self._offsets.append(self._offsets[-1] + len(d))
        logging.info(f"[dataset_mix] combined size: {self._offsets[-1]}")

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getitem__(self, idx: int):
        # binary search-ish linear since N is small (few grounding datasets)
        for i in range(len(self._datasets)):
            if idx < self._offsets[i + 1]:
                return self._datasets[i][idx - self._offsets[i]]
        raise IndexError(idx)


class _EpisodeSubsetWrapper(torch.utils.data.Dataset):  # type: ignore[misc]
    """Filters a LeRobotDataset to only include frames from specified episodes.

    Reads episode_index directly from the Arrow table (not metadata) to
    build the valid frame index list.
    """

    def __init__(self, base, episode_subset: set[int]):
        self._base = base
        self.meta = getattr(base, "meta", None)
        ep_col = base.hf_dataset.data.column("episode_index").to_pylist()
        self._valid = [i for i, ep in enumerate(ep_col) if int(ep) in episode_subset]
        logging.info(f"[episode_subset] {len(self._valid)} frames from {len(episode_subset)} episodes")

    def __len__(self) -> int:
        return len(self._valid)

    def __getitem__(self, idx: int):
        return self._base[self._valid[idx]]


class _DexGroundingBalancedDataset(torch.utils.data.Dataset):  # type: ignore[misc]
    """Strided N:1 interleaving of drag and grounding samples."""

    def __init__(
        self,
        dex: torch.utils.data.Dataset,
        grounding: torch.utils.data.Dataset,
        meta,
        dex_time_weights: list[float],
        grounding_time_weights: list[float],
        dex_grounding_ratio: int = 1,
    ):
        self.dex = dex
        self.grounding = grounding
        self.meta = meta
        self._dex_len = len(dex)
        self._grounding_len = len(grounding)
        if self._dex_len <= 0:
            raise ValueError("drag dataset is empty.")
        if self._grounding_len <= 0:
            raise ValueError("grounding dataset is empty.")
        self._ratio = max(1, int(dex_grounding_ratio))
        self._stride = self._ratio + 1  # e.g. ratio=2 => stride=3: dex,dex,grounding
        self._dex_time_weights = torch.tensor(dex_time_weights, dtype=torch.float32)
        self._grounding_time_weights = torch.tensor(grounding_time_weights, dtype=torch.float32)
        self._bbox_norm_placeholder = torch.tensor([-1.0, -1.0, -1.0, -1.0], dtype=torch.float32)

    def __len__(self) -> int:
        # Ensure both datasets are fully covered across the strided index space.
        dex_groups = (self._dex_len + self._ratio - 1) // self._ratio
        groups = max(dex_groups, self._grounding_len)
        return groups * self._stride

    def __getitem__(self, idx: int):
        pos_in_stride = idx % self._stride
        group = idx // self._stride
        if pos_in_stride < self._ratio:
            # dex sample — (group*ratio + pos) wraps around dex_len
            dex_idx = (group * self._ratio + pos_in_stride) % self._dex_len
            item = self.dex[dex_idx]
            item["loss_time_weights"] = self._dex_time_weights
            item["bbox_norm"] = self._bbox_norm_placeholder
            item["is_grounding"] = torch.tensor(False)
            return item
        # grounding sample — group wraps around grounding_len
        item = self.grounding[group % self._grounding_len]
        item["loss_time_weights"] = self._grounding_time_weights
        item["is_grounding"] = torch.tensor(True)
        return item


class _WithLossTimeWeightsDataset(torch.utils.data.Dataset):  # type: ignore[misc]
    def __init__(self, base: torch.utils.data.Dataset, meta, time_weights: list[float], is_grounding: bool = False):
        self.base = base
        self.meta = meta
        self._time_weights = torch.tensor(time_weights, dtype=torch.float32)
        self._is_grounding = torch.tensor(is_grounding)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        item["loss_time_weights"] = self._time_weights
        if "is_grounding" not in item:
            item["is_grounding"] = self._is_grounding
        return item

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if not isinstance(cfg.dataset.repo_id, str):
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")

    mode = getattr(cfg.dataset, "mode", "dex")
    if mode not in ("dex", "grounding", "dex+grounding"):
        raise ValueError(f"Unsupported dataset.mode='{mode}', expected one of 'dex', 'grounding', 'dex+grounding'.")

    filter_portrait = bool(getattr(cfg.dataset, "filter_portrait", False))

    drag_time_weights = [1.0] * int(cfg.policy.chunk_size)
    for i in range(min(5, len(drag_time_weights))):
        drag_time_weights[i] = 5.0
    for i in range(max(0, len(drag_time_weights) - 5), len(drag_time_weights)):
        drag_time_weights[i] = 5.0

    # Mode: drag-only (no grounding datasets).
    if mode == "dex":
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        if not cfg.dataset.streaming:
            base_ds = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
            )
        else:
            base_ds = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
            )
        dataset = _WithLossTimeWeightsDataset(base_ds, ds_meta, drag_time_weights, is_grounding=False)

    # Mode: grounding-only (no drag frames).
    elif mode == "grounding":
        if not cfg.dataset.grounding_datasets:
            raise ValueError("dataset.grounding_datasets must not be empty when mode='grounding'.")
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )

        base_ds: LeRobotDataset | StreamingLeRobotDataset | None = None
        online_datasets: list[torch.utils.data.Dataset] = []
        for name in cfg.dataset.grounding_datasets:
            if name == "waveui":
                online_datasets.append(
                    WaveUIOnlineDataset(
                        data_path=cfg.dataset.data_paths.wave_ui_json,
                        images_root=Path(cfg.dataset.data_paths.wave_ui_images),
                        chunk_size=cfg.policy.chunk_size,
                        filter_portrait=filter_portrait,
                    )
                )
            elif name == "guiact":
                online_datasets.append(
                    GUIActOnlineDataset(
                        data_path=cfg.dataset.data_paths.guiact_json,
                        images_root=Path(cfg.dataset.data_paths.guiact_images),
                        chunk_size=cfg.policy.chunk_size,
                        filter_portrait=filter_portrait,
                    )
                )
            elif name == "uground":
                online_datasets.append(
                    UgroundOnlineDataset(
                        data_path=cfg.dataset.data_paths.uground_json,
                        images_root=Path(cfg.dataset.data_paths.uground_images),
                        chunk_size=cfg.policy.chunk_size,
                        filter_portrait=filter_portrait,
                    )
                )
            elif name == "showui":
                online_datasets.append(
                    ShowUIOnlineDataset(
                        data_path=cfg.dataset.data_paths.showui_json,
                        images_root=Path(cfg.dataset.data_paths.showui_images),
                        chunk_size=cfg.policy.chunk_size,
                        filter_portrait=filter_portrait,
                    )
                )
            elif name == "segui":
                online_datasets.append(
                    SEGUIOnlineDataset(
                        data_path=cfg.dataset.data_paths.segui_json,
                        images_root=Path(cfg.dataset.data_paths.segui_images),
                        chunk_size=cfg.policy.chunk_size,
                        filter_portrait=filter_portrait,
                    )
                )
            else:
                raise ValueError(
                    f"Unknown grounding dataset: {name!r}. "
                    "Supported: waveui, guiact, uground, showui, segui."
                )

        dataset = _GroundingConcatDataset(base_ds, online_datasets, ds_meta)
        if cfg.policy.time_weights is None:
            raise ValueError("policy.time_weights is required when mode='grounding'.")
        if len(cfg.policy.time_weights) != int(cfg.policy.chunk_size):
            raise ValueError("len(policy.time_weights) must equal policy.chunk_size.")
        dataset = _WithLossTimeWeightsDataset(dataset, ds_meta, list(cfg.policy.time_weights), is_grounding=True)

    # Mode: drag + grounding combined.
    elif mode == "dex+grounding":
        if cfg.dataset.root is None:
            raise ValueError("dataset.root is required when mode='dex+grounding'.")

        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

        all_eps = ds_meta.episodes

        # Episode subset filtering (from JSON file)
        subset_json = getattr(cfg.dataset, "episode_subset_json", None)
        if subset_json:
            with open(subset_json) as _f:
                subset_set = set(json.load(_f))
            logging.info(f"[episode_subset] filtering to {len(subset_set)} episodes from {subset_json}")
        else:
            subset_set = None

        if cfg.dataset.episodes:
            allowed = set(int(e) for e in cfg.dataset.episodes)
            base_eps = [
                int(ep["episode_index"])
                for ep in all_eps
                if (int(ep["episode_index"]) in allowed and ep["length"] > 2)
            ]
        else:
            base_eps = [int(ep["episode_index"]) for ep in all_eps if ep["length"] > 2]

        if subset_set is not None:
            base_eps = [e for e in base_eps if e in subset_set]
            logging.info(f"[episode_subset] {len(base_eps)} episodes after filtering")

        if base_eps:
            if cfg.dataset.streaming:
                base_ds = StreamingLeRobotDataset(
                    cfg.dataset.repo_id,
                    root=cfg.dataset.root,
                    episodes=base_eps,
                    delta_timestamps=delta_timestamps,
                    image_transforms=image_transforms,
                    revision=cfg.dataset.revision,
                    max_num_shards=cfg.num_workers,
                )
            else:
                base_ds = LeRobotDataset(
                    cfg.dataset.repo_id,
                    root=cfg.dataset.root,
                    episodes=base_eps,
                    delta_timestamps=delta_timestamps,
                    image_transforms=image_transforms,
                    revision=cfg.dataset.revision,
                    video_backend=cfg.dataset.video_backend,
                )
        else:
            base_ds = None

        # Optional episode subset filter
        if base_ds is not None and subset_set is not None:
            base_ds = _EpisodeSubsetWrapper(base_ds, subset_set)

        online_datasets: list[torch.utils.data.Dataset] = []
        for name in cfg.dataset.grounding_datasets:
            if name == "waveui":
                online_datasets.append(
                    WaveUIOnlineDataset(
                        data_path=cfg.dataset.data_paths.wave_ui_json,
                        images_root=Path(cfg.dataset.data_paths.wave_ui_images),
                        chunk_size=cfg.policy.chunk_size,
                        filter_portrait=filter_portrait,
                    )
                )
            elif name == "guiact":
                online_datasets.append(
                    GUIActOnlineDataset(
                        data_path=cfg.dataset.data_paths.guiact_json,
                        images_root=Path(cfg.dataset.data_paths.guiact_images),
                        chunk_size=cfg.policy.chunk_size,
                        filter_portrait=filter_portrait,
                    )
                )
            elif name == "uground":
                online_datasets.append(
                    UgroundOnlineDataset(
                        data_path=cfg.dataset.data_paths.uground_json,
                        images_root=Path(cfg.dataset.data_paths.uground_images),
                        chunk_size=cfg.policy.chunk_size,
                        filter_portrait=filter_portrait,
                    )
                )
            elif name == "showui":
                online_datasets.append(
                    ShowUIOnlineDataset(
                        data_path=cfg.dataset.data_paths.showui_json,
                        images_root=Path(cfg.dataset.data_paths.showui_images),
                        chunk_size=cfg.policy.chunk_size,
                        filter_portrait=filter_portrait,
                    )
                )
            elif name == "segui":
                online_datasets.append(
                    SEGUIOnlineDataset(
                        data_path=cfg.dataset.data_paths.segui_json,
                        images_root=Path(cfg.dataset.data_paths.segui_images),
                        chunk_size=cfg.policy.chunk_size,
                        filter_portrait=filter_portrait,
                    )
                )
            else:
                raise ValueError(
                    f"Unknown grounding dataset: {name!r}. "
                    "Supported: waveui, guiact, uground, showui, segui."
                )

        if base_ds is None:
            raise ValueError("no valid episodes found (length > 2) for mode='dex+grounding'.")
        # Build grounding time_weights: only first 2 steps get loss, rest = 0
        chunk = int(cfg.policy.chunk_size)
        if cfg.policy.time_weights is not None:
            if len(cfg.policy.time_weights) != chunk:
                raise ValueError("len(policy.time_weights) must equal policy.chunk_size.")
            grounding_tw = list(cfg.policy.time_weights)
        else:
            grounding_tw = [0.0] * chunk
            grounding_tw[0] = 10.0
            if chunk > 1:
                grounding_tw[1] = 1.0
        dex_grounding_ratio = int(getattr(cfg.policy, "dex_grounding_ratio", 1))
        grounding_ds = _ConcatDataset(online_datasets, ds_meta)
        dataset = _DexGroundingBalancedDataset(
            base_ds,
            grounding_ds,
            ds_meta,
            dex_time_weights=drag_time_weights,
            grounding_time_weights=grounding_tw,
            dex_grounding_ratio=dex_grounding_ratio,
        )

    if cfg.dataset.use_imagenet_stats:
        meta = getattr(dataset, "meta", None)
        if meta is not None:
            for key in meta.camera_keys:
                for stats_type, stats in IMAGENET_STATS.items():
                    meta.stats.setdefault(key, {})[stats_type] = torch.tensor(
                        stats, dtype=torch.float32
                    )

    return dataset
