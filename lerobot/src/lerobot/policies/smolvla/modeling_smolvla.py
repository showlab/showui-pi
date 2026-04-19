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

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel, apply_lora_to_model
from lerobot.policies.utils import (
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.utils import get_safe_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"

    def __init__(
        self,
        config: SmolVLAConfig,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = VLAFlowMatching(config)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch, noise)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def forward(
        self,
        batch: dict[str, Tensor],
        noise=None,
        time=None,
        reflow_noise: Tensor | None = None,
        reflow_actions: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss.

        Args:
            batch: training batch from dataloader
            noise: optional noise tensor (if None, will be sampled)
            time: optional time tensor (if None, will be sampled)
            reflow_noise: for reflow training, the fixed noise that was used to generate reflow_actions
            reflow_actions: for reflow training, model-generated actions to train on (straighter paths)

        When reflow_noise and reflow_actions are provided, we train on the (noise, generated_action) pair
        instead of (random_noise, dataset_action). This straightens the flow paths (Rectified Flow).
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        # State noise augmentation: add noise to observation.state to simulate
        # closed-loop covariate shift (model seeing its own noisy predictions)
        if self.training and getattr(self.config, "state_noise_std", 0.0) > 0:
            # state is padded to max_state_dim=32. Only dims [1,2] are real xy;
            # dim 0 is binary button (don't corrupt); dims 3..31 are pad zeros
            # that state_proj must keep seeing as zero so train == eval distribution.
            noise_xy = torch.randn(
                state.shape[0], 2, device=state.device, dtype=state.dtype
            ) * self.config.state_noise_std
            state = state.clone()
            state[:, 1:3] = state[:, 1:3] + noise_xy
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        # Determine if we're in reflow mode
        is_reflow = reflow_noise is not None and reflow_actions is not None

        if is_reflow:
            # Reflow mode: use provided (noise, action) pair
            actions = reflow_actions  # Already padded to max_action_dim from generate_reflow_targets
            noise = reflow_noise
        else:
            # Normal mode: use dataset actions
            actions = self.prepare_action(batch)

        actions_is_pad = batch.get("action_is_pad")
        if actions_is_pad is None:
            raise ValueError("batch is missing action_is_pad")
        loss_time_weights = batch.get("loss_time_weights")
        if loss_time_weights is None:
            raise ValueError("batch is missing loss_time_weights")
        if loss_time_weights.shape != actions_is_pad.shape:
            raise ValueError("loss_time_weights shape must match action_is_pad")

        # Per-sample is_grounding flag for conditional lambda_dir
        is_grounding = batch.get("is_grounding")  # (B,) bool or None

        # Get original action for comparison
        # In reflow mode, we still compare against dataset actions for pixel error metrics
        # but the FM loss uses the reflow_actions
        actions_original = batch[ACTION]  # (B, T, original_dim) e.g., (B, T, 3)
        original_action_dim = self.config.action_feature.shape[0]  # e.g., 3

        loss_dict = {}
        losses, mse_term, l1_term, cos_term, a_hat, x_t, time_expanded = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time,
            is_grounding=is_grounding,
        )

        # Crop losses to real action dimensions only.
        losses = losses[..., :original_action_dim]
        mse_term = mse_term[..., :original_action_dim]
        l1_term = l1_term[..., :original_action_dim]
        cos_term = cos_term[..., :original_action_dim]

        # Mask out-of-episode (padded) timesteps before any aggregation
        in_episode_bound = ~actions_is_pad  # (B, T) True where valid
        mask = in_episode_bound.unsqueeze(-1)  # (B, T, 1)
        w_t = loss_time_weights.to(dtype=losses.dtype).unsqueeze(-1)  # (B, T, 1)
        losses = losses * mask * w_t
        mse_term = mse_term * mask * w_t
        l1_term = l1_term * mask * w_t
        cos_term = cos_term * mask * w_t
        weight_sum = (loss_time_weights.to(dtype=losses.dtype) * in_episode_bound.to(dtype=losses.dtype)).sum(dim=1)
        weight_sum = weight_sum.clamp(min=1e-8)  # avoid division by zero for fully-padded samples

        # Action reconstruction loss (a_hat = x_t - t*v_t).
        # a_hat: (B, T, 32) predicted action, crop to original_dim
        a_hat_cropped = a_hat[..., :original_action_dim]  # (B, T, 3)

        # Compute action reconstruction error on x,y coordinates (indices 1,2)
        # For grounding: action = [btn, x, y] where x,y are in [0,1]
        action_recon_err = a_hat_cropped - actions_original  # (B, T, 3)

        xy_err = action_recon_err[..., 1:3]  # (B, T, 2) for x,y

        # Action reconstruction loss
        lambda_action_recon = getattr(self.config, "lambda_action_recon", 1.0)
        action_recon_loss_per_elem = action_recon_err**2  # (B, T, 3)
        action_recon_loss_per_elem = action_recon_loss_per_elem * mask * w_t

        # Aggregate action reconstruction loss
        num_action_recon = action_recon_loss_per_elem.sum(dim=(1, 2))  # (B,)

        # ===== Compute pixel errors for monitoring (only on x,y) =====
        # x,y are normalized to [0,1], so convert to pixels
        # Only compute pixel error on timesteps with positive time weight
        mask_f = mask.to(dtype=xy_err.dtype)
        w_mask = (loss_time_weights > 0).to(dtype=xy_err.dtype).unsqueeze(-1)  # (B, T, 1)
        xy_err_for_metric = xy_err * mask_f * w_mask
        num_valid_metric = (mask_f * w_mask).sum()
        pixel_err_x = (xy_err_for_metric[..., 0].abs() * 1920.0).sum() / num_valid_metric
        pixel_err_y = (xy_err_for_metric[..., 1].abs() * 1080.0).sum() / num_valid_metric
        dx_px = xy_err_for_metric[..., 0] * 1920.0
        dy_px = xy_err_for_metric[..., 1] * 1080.0
        pixel_err_avg = torch.sqrt(dx_px * dx_px + dy_px * dy_px).sum() / num_valid_metric

        # ===== Normalize losses by effective timesteps and dimensions =====
        loss_action_dim = losses.shape[2]
        den = weight_sum * loss_action_dim

        num_total = losses.sum(dim=(1, 2))
        num_mse = mse_term.sum(dim=(1, 2))
        num_l1 = l1_term.sum(dim=(1, 2))
        num_cos = cos_term.sum(dim=(1, 2))

        # FM loss: normalized by effective timesteps.
        loss_fm = (num_total / den).mean()
        loss_action_recon = (num_action_recon / den).mean() * lambda_action_recon

        # Total loss = FM loss + action reconstruction loss
        loss = loss_fm + loss_action_recon

        mse_scalar = (num_mse / den).mean().item()
        l1_scalar = (num_l1 / den).mean().item()
        cos_scalar = (num_cos / den).mean().item()

        # For backward pass and logging
        loss_dict["loss"] = loss.item()
        loss_dict["loss_fm"] = loss_fm.item()
        loss_dict["loss_action_recon"] = loss_action_recon.item()
        loss_dict["loss_mse"] = mse_scalar
        loss_dict["loss_l1"] = l1_scalar
        loss_dict["loss_cos"] = cos_scalar
        loss_dict["pixel_err_x"] = pixel_err_x.item()
        loss_dict["pixel_err_y"] = pixel_err_y.item()
        loss_dict["pixel_err_avg"] = pixel_err_avg.item()
        loss_dict["is_reflow"] = 1.0 if is_reflow else 0.0

        return loss, loss_dict

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            vision_scale_factor=self.config.vision_scale_factor,
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        self.set_requires_grad()

        # Apply LoRA to VLM text model if configured
        if getattr(self.config, "use_lora", False):
            rank = getattr(self.config, "lora_rank", 16)
            alpha = getattr(self.config, "lora_alpha", 32)
            dropout = getattr(self.config, "lora_dropout", 0.05)
            vlm_text_model = self.vlm_with_expert.get_vlm_model().text_model
            apply_lora_to_model(vlm_text_model, rank=rank, alpha=alpha, dropout=dropout,
                                target_modules=("q_proj", "v_proj"))
            # Freeze all non-LoRA VLM params (base weights already frozen by LoRALinear,
            # but also freeze embeddings, norms, etc.)
            for name, param in self.vlm_with_expert.vlm.named_parameters():
                if "lora_" not in name:
                    param.requires_grad = False

        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device, is_grounding=None):
        """Sample noise for FM. Uses per-task noise_std if noise_std_dex is set."""
        std_default = float(self.config.noise_std)
        std_dex = getattr(self.config, "noise_std_dex", None)

        if is_grounding is not None and std_dex is not None:
            std_dex = float(std_dex)
            # Per-sample noise std: dex uses std_dex, grounding uses noise_std
            noise = torch.randn(shape, dtype=torch.float32, device=device)
            gr_mask = is_grounding.to(device=device)  # (B,)
            # Expand to (B, 1, 1) for broadcasting against (B, T, A)
            std_per_sample = torch.where(gr_mask, std_default, std_dex).view(-1, 1, 1)
            noise = noise * std_per_sample
        else:
            noise = torch.normal(
                mean=0.0, std=std_default, size=shape,
                dtype=torch.float32, device=device,
            )

        action_dim = self.config.action_feature.shape[0]
        if shape[-1] < action_dim:
            raise ValueError(f"max_action_dim ({shape[-1]}) < action_dim ({action_dim})")
        if shape[-1] > action_dim:
            noise[..., action_dim:] = 0.0
        return noise

    def sample_time(self, bsize, device, is_grounding=None):
        tmin = float(self.config.time_min)
        if is_grounding is None:
            # Fallback: single Beta distribution (dex defaults)
            a, b = getattr(self.config, "time_beta_dex", (1.5, 1.0))
            beta_dist = torch.distributions.Beta(concentration1=a, concentration0=b)
            time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        else:
            # Per-sample: grounding gets Beta biased toward t→0, dex gets original
            a_dex, b_dex = getattr(self.config, "time_beta_dex", (1.5, 1.0))
            a_gr, b_gr = getattr(self.config, "time_beta_grounding", (0.5, 1.5))
            dist_dex = torch.distributions.Beta(a_dex, b_dex)
            dist_gr = torch.distributions.Beta(a_gr, b_gr)
            t_dex = dist_dex.sample((bsize,)).to(device=device, dtype=torch.float32)
            t_gr = dist_gr.sample((bsize,)).to(device=device, dtype=torch.float32)
            gr_mask = is_grounding.to(device=device)
            time_beta = torch.where(gr_mask, t_gr, t_dex)
        time = time_beta * (1.0 - tmin) + tmin
        return time

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state.to(self.state_proj.weight.dtype))
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        noisy_actions = noisy_actions.to(self.action_in_proj.weight.dtype)
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None,
        is_grounding=None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)

        Returns:
            losses: (B, T, A) FM loss (MSE + L1)
            mse_term: (B, T, A) MSE component
            l1_term: (B, T, A) L1 component on traj_reg_indices
            cos_term: (B, T, A) directional penalty term
            a_hat: (B, T, A) predicted action via x_t - t*v_t
            x_t: (B, T, A) noised action
            time_expanded: (B, 1, 1) sampled time
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device, is_grounding=is_grounding)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device, is_grounding=is_grounding)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # x_t conditioning dropout: zero out x_t for random samples to force image/text reliance
        xt_drop_p = getattr(self.config, "xt_dropout", 0.0)
        if self.training and xt_drop_p > 0.0:
            drop_mask = (torch.rand(actions.shape[0], device=actions.device) < xt_drop_p)
            x_t = torch.where(drop_mask[:, None, None], torch.zeros_like(x_t), x_t)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Project to action dim, then upcast to float32 for loss computation
        v_t = self.action_out_proj(suffix_out)
        v_t = v_t.to(dtype=torch.float32)

        # Base FM components before any regularization:
        # Standard CFM loss: ||u_t - v_t||^2 (no time weighting)
        diff = u_t - v_t
        mse_term = diff * diff  # (B, T, A)
        l1_term = torch.zeros_like(mse_term)
        cos_term = torch.zeros_like(mse_term)

        # Add L1 term on dimensions specified by traj_reg_indices.
        if getattr(self.config, "traj_reg_indices", None):
            idxs = self.config.traj_reg_indices
            l1 = diff[:, :, idxs].abs()
            l1_term[:, :, idxs] = l1

        # Total loss: MSE + L1 (on selected dimensions).
        losses = mse_term + l1_term

        # Optional directional alignment penalty on specified dims (e.g., (x,y)).
        # Only applied to drag samples (is_grounding=False); grounding clicks have no
        # meaningful trajectory direction.
        if (
            getattr(self.config, "lambda_dir", 0.0) > 0.0
            and getattr(self.config, "traj_reg_indices", None)
        ):
            idxs = self.config.traj_reg_indices
            v2 = v_t[:, :, idxs]
            u2 = u_t[:, :, idxs]
            # cos_sim across last-dim (selected dims)
            eps = torch.finfo(v2.dtype).eps
            vnorm = torch.linalg.norm(v2, dim=-1)
            unorm = torch.linalg.norm(u2, dim=-1)
            denom = (vnorm * unorm + eps)
            cos = torch.sum(v2 * u2, dim=-1) / denom
            dir_pen = 1.0 - cos  # (B, T) — uniform in time for regularizer
            # Per-sample masking: zero out dir penalty for grounding samples
            if is_grounding is not None:
                drag_mask = (~is_grounding).to(dtype=dir_pen.dtype)  # (B,) 1 for drag, 0 for grounding
                dir_pen = dir_pen * drag_mask[:, None]  # (B, T)
            # Broadcast to selected dims and add to losses / cos_term
            cos_add = self.config.lambda_dir * dir_pen[:, :, None]
            cos_term[:, :, idxs] = cos_add
            losses[:, :, idxs] = losses[:, :, idxs] + cos_add

        # Add optional second-order smoothness regularization on specified action dims
        # Only for drag samples (same rationale as lambda_dir).
        if (
            getattr(self.config, "lambda_traj_acc", 0.0) > 0.0
            and getattr(self.config, "traj_reg_indices", None)
            and v_t.shape[1] >= 3
        ):
            idxs = self.config.traj_reg_indices
            d2 = v_t[:, 2:, idxs] - 2 * v_t[:, 1:-1, idxs] + v_t[:, :-2, idxs]
            reg = (d2 * d2)  # uniform in time for regularizer
            if is_grounding is not None:
                drag_mask = (~is_grounding).to(dtype=reg.dtype)
                reg = reg * drag_mask[:, None, None]
            losses[:, 1:-1, idxs] = losses[:, 1:-1, idxs] + self.config.lambda_traj_acc * reg

        # Compute predicted action via x_t - t*v_t (for action reconstruction loss and monitoring)
        a_hat = x_t - time_expanded * v_t

        return losses, mse_term, l1_term, cos_term, a_hat, x_t, time_expanded

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        n_ensemble = getattr(self.config, "inference_ensemble", 1)
        if n_ensemble <= 1:
            return self._ode_solve(bsize, device, prefix_pad_masks, past_key_values, noise)
        else:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            results = []
            for _ in range(n_ensemble):
                n = self.sample_noise(actions_shape, device)
                results.append(self._ode_solve(bsize, device, prefix_pad_masks, past_key_values, n))
            return torch.stack(results, dim=0).median(dim=0).values

    def _ode_solve(self, bsize, device, prefix_pad_masks, past_key_values, noise=None):
        """Run ODE integration from t=1 to t=0.

        Inference modes controlled by config:
        - inference_zero_noise: start from x_1=0 instead of random noise (deterministic)
        - inference_ahat_median: return median of a_hat predictions across ODE steps
        """
        use_zero_noise = getattr(self.config, "inference_zero_noise", False)
        use_ahat_median = getattr(self.config, "inference_ahat_median", False)

        if use_zero_noise:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            x_t = torch.zeros(actions_shape, dtype=torch.float32, device=device)
        elif noise is not None:
            x_t = noise
        else:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            x_t = self.sample_noise(actions_shape, device)

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        a_hats = []
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            if use_ahat_median:
                # Collect direct action prediction: a_hat = x_t - t * v_t
                a_hat = x_t - time * v_t
                a_hats.append(a_hat)
            # Euler step
            x_t += dt * v_t
            time += dt

        if use_ahat_median and len(a_hats) > 0:
            return torch.stack(a_hats, dim=0).median(dim=0).values
        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        v_t = self.action_out_proj(suffix_out)
        v_t = v_t.to(dtype=torch.float32)
        return v_t

    @torch.no_grad()
    def generate_reflow_targets(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        num_steps: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Generate (noise, action) pairs for reflow training.

        Instead of using dataset actions, we generate actions from random noise
        using the current model. This creates deterministic (noise, action) pairs
        that lead to straighter flow paths when used for training.

        Args:
            images, img_masks, lang_tokens, lang_masks, state: conditioning inputs
            num_steps: ODE solver steps (defaults to config.reflow_num_steps)

        Returns:
            noise: the starting noise (B, T, A)
            generated_actions: model-generated actions from that noise (B, T, A)
        """
        bsize = state.shape[0]
        device = state.device

        # Sample random noise - this will be our "starting point"
        actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
        noise = self.sample_noise(actions_shape, device)

        # Use specified num_steps or fall back to config
        if num_steps is None:
            num_steps = getattr(self.config, "reflow_num_steps", self.config.num_steps)

        # Generate actions via ODE solving (similar to sample_actions but with custom num_steps)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Cache prefix KV
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        # ODE solve from noise to action
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise.clone()  # Start from noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            # Euler step
            x_t += dt * v_t
            time += dt

        generated_actions = x_t  # Final denoised actions

        return noise, generated_actions
