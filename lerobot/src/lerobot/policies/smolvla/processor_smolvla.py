#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    EnvTransition,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    TransitionKey,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import OBS_STATE, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def make_smolvla_pre_post_processors(
    config: SmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the SmolVLA policy.

    The pre-processing pipeline prepares input data for the model by:
    1.  Renaming features to match pretrained configurations.
    2.  Adding a batch dimension.
    3.  Ensuring the language task description ends with a newline character.
    4.  Tokenizing the language task description.
    5.  Moving all data to the specified device.
    6.  Normalizing (x,y) from 1920×1080 pixels to [0,1] (x<0 or y<0 keeps unchanged).

    The post-processing pipeline handles the model's output by:
    1.  Unnormalizing (x,y) from [0,1] back to 1920×1080 pixels.
    2.  Moving data to the CPU.

    Args:
        config: The configuration object for the SmolVLA policy.
        dataset_stats: Unused (kept for API compatibility).

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        SmolVLANewLineProcessor(),
        TokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
        DeviceProcessorStep(device=config.device),
        SmolVLAFixedXYNormalizerProcessorStep(),
    ]
    output_steps = [
        SmolVLAFixedXYUnnormalizerProcessorStep(),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


@ProcessorStepRegistry.register(name="smolvla_fixed_xy_normalizer")
@dataclass
class SmolVLAFixedXYNormalizerProcessorStep(ProcessorStep):
    """Normalize (x,y) from pixel to [0,1] in 1920×1080; keep x<0 or y<0 unchanged."""

    width: float = 1920.0
    height: float = 1080.0

    def get_config(self) -> dict[str, Any]:
        return {"width": float(self.width), "height": float(self.height)}

    def _norm_xy(self, t: torch.Tensor) -> torch.Tensor:
        if t.shape[-1] < 3:
            return t
        out = t.clone()
        x = out[..., 1]
        y = out[..., 2]
        out[..., 1] = torch.where(x >= 0.0, x / self.width, x)
        out[..., 2] = torch.where(y >= 0.0, y / self.height, y)
        return out

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()

        obs = new_transition.get(TransitionKey.OBSERVATION.value)
        if isinstance(obs, dict) and OBS_STATE in obs:
            obs = dict(obs)
            v = obs[OBS_STATE]
            if isinstance(v, torch.Tensor):
                obs[OBS_STATE] = self._norm_xy(v)
            new_transition[TransitionKey.OBSERVATION.value] = obs

        act = new_transition.get(TransitionKey.ACTION.value)
        if isinstance(act, torch.Tensor):
            new_transition[TransitionKey.ACTION.value] = self._norm_xy(act)

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="smolvla_fixed_xy_unnormalizer")
@dataclass
class SmolVLAFixedXYUnnormalizerProcessorStep(ProcessorStep):
    """Unnormalize (x,y) from [0,1] back to 1920×1080 pixels; keep x<0 or y<0 unchanged."""

    width: float = 1920.0
    height: float = 1080.0

    def get_config(self) -> dict[str, Any]:
        return {"width": float(self.width), "height": float(self.height)}

    def _unnorm_xy(self, t: torch.Tensor) -> torch.Tensor:
        if t.shape[-1] < 3:
            return t
        out = t.clone()
        x = out[..., 1]
        y = out[..., 2]
        out[..., 1] = torch.where(x >= 0.0, x * self.width, x)
        out[..., 2] = torch.where(y >= 0.0, y * self.height, y)
        return out

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        act = new_transition.get(TransitionKey.ACTION.value)
        if isinstance(act, torch.Tensor):
            new_transition[TransitionKey.ACTION.value] = self._unnorm_xy(act)
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="smolvla_new_line_processor")
class SmolVLANewLineProcessor(ComplementaryDataProcessorStep):
    """
    A processor step that ensures the 'task' description ends with a newline character.

    This step is necessary for certain tokenizers (e.g., PaliGemma) that expect a
    newline at the end of the prompt. It handles both single string tasks and lists
    of string tasks.
    """

    def complementary_data(self, complementary_data):
        if "task" not in complementary_data:
            return complementary_data

        task = complementary_data["task"]
        if task is None:
            return complementary_data

        new_complementary_data = dict(complementary_data)

        # Handle both string and list of strings
        if isinstance(task, str):
            # Single string: add newline if not present
            if not task.endswith("\n"):
                new_complementary_data["task"] = f"{task}\n"
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            # List of strings: add newline to each if not present
            new_complementary_data["task"] = [t if t.endswith("\n") else f"{t}\n" for t in task]
        # If task is neither string nor list of strings, leave unchanged

        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
