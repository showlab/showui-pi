# ShowUI-&pi;

Open-source, End-to-end, Lightweight, Vision-Language-Action Model for GUI Drag operations.

ShowUI-π 是一款开源的、端到端、轻量级的视觉-语言-动作模型，专为 GUI 拖拽交互设计。

<p align="center">
        &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2512.24965">Paper</a> &nbsp&nbsp 
        | 🤗 <a href="https://huggingface.co/showlab/ShowUI-pi">Model</a>&nbsp&nbsp 
        | &nbsp&nbsp 🤗 <a href="https://huggingface.co/datasets/h-siyuan/ScreenDrag">Datasets</a> &nbsp&nbsp 
</p>

> [**ShowUI-&pi;: Flow-based Generative Models as GUI Dexterous Hands**](https://arxiv.org/abs/2512.24965)<br>
> [Siyuan Hu](https://sy-h.com/)\*, [Kevin Qinghong Lin](https://qhlin.me/)\*, [Mike Zheng Shou](https://scholar.google.com/citations?user=h1-3lSoAAAAJ&hl=en)
> <br>Show Lab @ National University of Singapore<br>

## 🔥 Update
- [x] [2026.2.20] **ShowUI-&pi;** is accepted by **CVPR 2026**.
- [x] [2025.12.31] We released [**ShowUI-&pi;**](https://github.com/showlab/showui-pi) for GUI dragging.
- [x] [2025.12.31] We released the [**DEX Benchmark**](https://huggingface.co/datasets/h-siyuan/ScreenDrag) for GUI drag-and-drop evaluation.

## ⭐ Quick Start

```bash
git clone https://github.com/showlab/showui-pi.git
cd showui-pi
pip install -e .
```

## 🚀 Training

```bash
bash scripts/train_showui_pi.sh
```

See [`scripts/train_showui_pi.sh`](scripts/train_showui_pi.sh) for all flags and defaults.

## 🕹️ Evaluation

### DEX Benchmark

The [DEX benchmark](https://huggingface.co/datasets/h-siyuan/ScreenDrag) is downloaded automatically on first run.

```bash
PYTHONPATH=lerobot/src \
python scripts/eval_dex.py \
    --ckpt <path/to/checkpoint> \
    --output_dir outputs/eval_dex
```

### ScreenSpot-Pro

```bash
PYTHONPATH=lerobot/src \
python scripts/eval_screenspot_pro.py \
    --ckpt <path/to/checkpoint> \
    --annotations_root <path/to/ScreenSpot-Pro/annotations> \
    --images_root <path/to/ScreenSpot-Pro/images>
```

## ❤ Acknowledgement

We extend our gratitude to [LeRobot](https://github.com/huggingface/lerobot) and [SmolVLA](https://huggingface.co/lerobot/smolvla_base) for the training framework, and [SeeClick](https://github.com/njucckevin/SeeClick) for grounding data.

## 🎓 BibTeX

```
@article{hu2025showui,
  title={ShowUI-$$\backslash$pi $: Flow-based Generative Models as GUI Dexterous Hands},
  author={Hu, Siyuan and Lin, Kevin Qinghong and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2512.24965},
  year={2025}
}
```

If you like our project, please give us a star ⭐ on GitHub for the latest update.

[![Star History Chart](https://api.star-history.com/svg?repos=showlab/showui-pi&type=Timeline&width=600&height=300)](https://star-history.com/#showlab/showui-pi&Timeline)
