#!/bin/bash
# showui-pi training: DEX drag + grounding (waveui, guiact, uground) via SmolVLA.
#
# Default: 8 GPUs, 15000 steps, bfloat16, DeepSpeed ZeRO-2.
# Data paths are controlled by --dataset.data_paths.<field>; defaults are
# defined in lerobot/src/lerobot/configs/default.py::DataPathConfig and
# documented in data/README.md.

set -euo pipefail

# Ensure the lerobot package (under lerobot/src) is importable.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/lerobot/src:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch \
  --num_processes 8 \
  --mixed_precision bf16 \
  --use_deepspeed \
  --zero_stage 2 \
  --offload_optimizer_device none \
  --offload_param_device none \
  --gradient_accumulation_steps 1 \
  --gradient_clipping 1.0 \
  --zero3_init_flag false \
  -m lerobot.scripts.lerobot_train_accelerate \
  --policy.type=smolvla \
  --policy.push_to_hub=false \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --policy.load_vlm_weights=true \
  --policy.resize_imgs_with_padding='[1024,576]' \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=false \
  --policy.use_lora=false \
  --policy.vision_scale_factor=4 \
  --policy.chunk_size=21 \
  --policy.n_action_steps=1 \
  --policy.traj_reg_indices='[1,2]' \
  --policy.lambda_dir=0.0 \
  --policy.lambda_action_recon=0.0 \
  --policy.dex_grounding_ratio=4 \
  --policy.max_action_dim=3 \
  --policy.num_steps=10 \
  --policy.noise_std=0.3 \
  --policy.state_noise_std=0.02 \
  --policy.optimizer_lr=5e-5 \
  --policy.scheduler_warmup_steps=0 \
  --policy.scheduler_decay_steps=15000 \
  --dataset.mode=dex+grounding \
  --dataset.repo_id=dex \
  --dataset.revision=v3.0 \
  --dataset.use_imagenet_stats=true \
  --dataset.grounding_datasets='["waveui","guiact","uground"]' \
  --dataset.filter_portrait=true \
  --batch_size=8 \
  --num_workers=4 \
  --steps=15000 \
  --seed=42 \
  --eval_freq=0 \
  --log_freq=10 \
  --save_freq=2500 \
  --job_name=showui_pi \
  --output_dir=outputs/showui_pi \
  --wandb.enable=true \
  --wandb.project=showui-pi
