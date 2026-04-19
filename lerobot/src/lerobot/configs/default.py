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

from dataclasses import dataclass, field

from lerobot import (
    policies,  # noqa: F401
)
from lerobot.datasets.transforms import ImageTransformsConfig
from lerobot.datasets.video_utils import get_safe_default_codec


@dataclass
class DataPathConfig:
    """Filesystem paths to DEX and grounding datasets.

    Each grounding dataset has two paths: a JSON annotations file and an
    images root directory. Defaults point to ``data/`` relative to the
    project root; override via CLI, e.g.
    ``--dataset.data_paths.wave_ui_json=/my/wave_ui.json``.

    The optional ``showui_*`` and ``segui_*`` fields are available for
    extensibility but are not used by the default training configuration.
    """
    dex_root: str = "data/dex"
    # WaveUI
    wave_ui_json: str = "data/wave_ui/annotations.json"
    wave_ui_images: str = "data/wave_ui/images"
    # GUIAct
    guiact_json: str = "data/guiact/annotations.json"
    guiact_images: str = "data/guiact/images"
    # UGround
    uground_json: str = "data/uground/annotations.json"
    uground_images: str = "data/uground/images"
    # ShowUI (optional)
    showui_json: str = "data/showui_desktop/annotations.json"
    showui_images: str = "data/showui_desktop/images"
    # SEGUI (optional)
    segui_json: str = "data/segui/annotations.json"
    segui_images: str = "data/segui/images"


@dataclass
class DatasetConfig:
    # You may provide a list of datasets here. `train.py` creates them all and concatenates them. Note: only data
    # keys common between the datasets are kept. Each dataset gets and additional transform that inserts the
    # "dataset_index" into the returned item. The index mapping is made according to the order in which the
    # datasets are provided.
    repo_id: str
    data_paths: DataPathConfig = field(default_factory=DataPathConfig)
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | None = None
    episodes: list[int] | None = None
    # Optional: load episode indices from an external file (each line: <root>\t<episode_index>).
    episodes_file: str | None = None
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = field(default_factory=get_safe_default_codec)
    streaming: bool = False
    # Grounding datasets to include (e.g. ["waveui", "guiact", "uground"]).
    grounding_datasets: list[str] = field(default_factory=list)
    # Data source mode: "dex", "grounding", or "dex+grounding".
    mode: str = "dex"
    # Filter out portrait screenshots (height > width).
    filter_portrait: bool = False
    # Optional: train on a subset of episodes (JSON file with episode_index list).
    episode_subset_json: str | None = None


@dataclass
class WandBConfig:
    enable: bool = False
    # Set to true to disable saving an artifact despite training.save_checkpoint=True
    disable_artifact: bool = False
    project: str = "lerobot"
    entity: str | None = None
    notes: str | None = None
    run_id: str | None = None
    mode: str | None = None  # Allowed values: 'online', 'offline' 'disabled'. Defaults to 'online'


@dataclass
class EvalConfig:
    n_episodes: int = 50
    # `batch_size` specifies the number of environments to use in a gym.vector.VectorEnv.
    batch_size: int = 50
    # `use_async_envs` specifies whether to use asynchronous environments (multiprocessing).
    use_async_envs: bool = False

    def __post_init__(self):
        if self.batch_size > self.n_episodes:
            raise ValueError(
                "The eval batch size is greater than the number of eval episodes "
                f"({self.batch_size} > {self.n_episodes}). As a result, {self.batch_size} "
                f"eval environments will be instantiated, but only {self.n_episodes} will be used. "
                "This might significantly slow down evaluation. To fix this, you should update your command "
                f"to increase the number of episodes to match the batch size (e.g. `eval.n_episodes={self.batch_size}`), "
                f"or lower the batch size (e.g. `eval.batch_size={self.n_episodes}`)."
            )
