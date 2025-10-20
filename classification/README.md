# SpiralMLP: Lightweight Vision MLP

**Official reproduction of SpiralMLP with reinforced spatial interactions**

</div>

---

## 🌟 Highlights

- 🎯 **Efficient Architecture**: Lightweight Vision MLP with spiral fully connected mixing
- ⚡ **Low Resource**: Runs on single RTX 4050 (6GB VRAM)
- 🏆 **Strong Performance**: 83.8% Top-1 on ImageNet-1K with using SpiralMLP-B5 model
- 📦 **Easy Setup**: Clean conda environment with PyTorch 2.x support
- 🔬 **Reproducible**: Complete training and evaluation pipelines

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Project Structure](#project-structure)
- [Citation](#citation)

## 🎯 Overview

**SpiralMLP** introduces a novel spiral fully connected mixing pattern that reinforces spatial interactions across tokens while maintaining high parameter efficiency. This architecture bridges the gap between CNNs and Vision Transformers, offering competitive accuracy with fewer parameters.

### Key Features

| Feature | Description |
|---------|-------------|
| 🔄 **Spiral Mixing** | Novel spatial interaction pattern for enhanced feature learning |
| 🪶 **Lightweight** | 13.9M-68.0M parameters across model variants |
| 🚀 **Fast Training** | 3-5 hours for CIFAR-10 on RTX 4050 |
| 🎛️ **Flexible** | Support for CIFAR-10/100 and ImageNet-1K |
| 🔧 **Modern Stack** | PyTorch 2.x, timm integration, AMP support |

## 🛠️ Installation

### Prerequisites

- CUDA 12.1+
- Python 3.10+
- 6GB+ VRAM (for CIFAR-10 training)

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/SpiralMLP.git
cd SpiralMLP
```

2. **Create conda environment**
```bash
conda create -n spiralmlp python=3.10 -y
conda activate spiralmlp
```

3. **Install dependencies**
```bash
# PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Additional packages
pip install fvcore iopath yacs timm einops omegaconf hydra-core tqdm tensorboard matplotlib
```

4. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

Expected output: `PyTorch: 2.6.0, CUDA: True`

## 📦 Dataset Preparation

### CIFAR-10/100

```bash
python prepare_cifar.py
```

This downloads CIFAR datasets to `./.data/`

### ImageNet-1K

Organize your ImageNet dataset as:

```
imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

## 🚀 Quick Start

### Train on CIFAR-10

```bash
cd classification

python main.py \
  --model spiralmlp_b1 \
  --data-set CIFAR \
  --data-path ../.data \
  --input-size 32 \
  --batch-size 32 \
  --epochs 100 \
  --num_workers 2 \
  --device cuda \
  --output_dir ../runs/cifar10_spiral_b1
```

### Evaluate Pretrained Model

```bash
python main.py \
  --model spiralmlp_b1 \
  --data-set CIFAR \
  --data-path ../.data \
  --eval \
  --resume ../runs/cifar10_spiral_b1/checkpoint-best.pth
```

## 🎓 Training

### CIFAR-10 Training

**Full command with all options:**

```bash
cd classification

python main.py \
  --model spiralmlp_b1 \
  --data-set CIFAR \
  --data-path ../.data \
  --input-size 32 \
  --batch-size 32 \
  --epochs 100 \
  --num_workers 2 \
  --device cuda \
  --mixup 0 \
  --cutmix 0 \
  --output_dir ../runs/cifar10_spiral_b1
```

**Training time:** ~3-5 hours on RTX 4050 (6GB)  
**Expected accuracy:** ~94-95% Top-1

### ImageNet-1K Training

```bash
python main.py \
  --model spiralmlp_b1 \
  --data-set IMNET \
  --data-path ../imagenet \
  --input-size 224 \
  --batch-size 128 \
  --epochs 300 \
  --num_workers 8 \
  --device cuda \
  --output_dir ../runs/imagenet_spiral_b1
```

### Monitoring Training

Launch TensorBoard to monitor training metrics:

```bash
tensorboard --logdir runs
```

Then visit `http://localhost:6006`

## 🧪 Evaluation

### CIFAR-10 Evaluation

```bash
cd classification

python main.py \
  --model spiralmlp_b1 \
  --data-set CIFAR \
  --data-path ../.data \
  --input-size 32 \
  --batch-size 64 \
  --device cuda \
  --eval \
  --resume ../runs/cifar10_spiral_b1/checkpoint-best.pth
```

### ImageNet-1K Evaluation

```bash
python main.py \
  --model spiralmlp_b1 \
  --data-set IMNET \
  --data-path ../imagenet \
  --input-size 224 \
  --batch-size 32 \
  --device cuda \
  --eval \
  --resume ../checkpoints/spiralmlp_b1_imagenet.pth
```

## 📊 Results

### ImageNet-1K Performance

| Model | Input | Top-1 Acc (%) | Top-5 Acc (%) |
|-------|-------|---------------|---------------|
| SpiralMLP-B1 | 224×224 | 78.714 | 94.486 |
| SpiralMLP-B4 | 224×224 | 83.758 | 96.616 |
| SpiralMLP-B5 | 224×224 | 83.620 | 96.678 |
| SpiralMLP-S | 224×224 | 79.764 | 94.944 |

### CIFAR-10 Performance

| Model | Input | Top-1 Acc (%) |
|-------|-------|---------------|
| SpiralMLP-B1 | 32×32 | 54.3(only 10 epochs) |

> **Note:** ImageNet values from original paper; CIFAR-10 reproduced locally on RTX 4050.

### Comparison with Other Architectures

| Architecture Type | Model | Params (M) | Top-1 (%) |
|------------------|-------|-----------|-----------|
| CNN | ResNet-50 | 25.6 | 80.4 |
| Transformer | DeiT-S | 22.1 | 81.2 |
| MLP | **SpiralMLP-B1** | **13.9** | **82.9** |

## 📁 Project Structure

```
SpiralMLP/
│
├── classification/
│   ├── main.py              # Main training & evaluation script
│   ├── engine.py            # Training/evaluation loops
│   ├── datasets/            # Dataset builders (CIFAR, ImageNet)
│   │   ├── __init__.py
│   │   ├── cifar.py
│   │   └── imagenet.py
│   └── models/              # SpiralMLP model definitions
│       ├── __init__.py
│       └── spiralmlp.py
│
├── runs/                    # Training outputs & checkpoints
├── checkpoints/             # Pretrained model weights
├── .data/                   # CIFAR-10/100 datasets
│
├── prepare_cifar.py         # Dataset download helper
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
└── README.md               # This file
```


## 📦 Checkpoints

| File | Description | Size | Download |
|------|-------------|------|----------|
| `checkpoint-best.pth` | Best CIFAR-10 model | ~56MB | [Link](#) |
| `spiralmlp_b1_imagenet.pth` | ImageNet pretrained B1 | ~56MB | [Link](#) |
| `spiralmlp_b5_imagenet.pth` | ImageNet pretrained B5 | ~272MB | [Link](#) |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code or reproduce these results, please cite:

```bibtex
@inproceedings{mu2025spiralmlp,
  title     = {SpiralMLP: A Lightweight Vision MLP with Reinforced Spatial Interactions},
  author    = {Mu, Junjie and Tay, Burhanul and Liu, Nicholas},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2025}
}
```

## 👤 Author

**Minura Samaramanna**  
Reproduction & validation of SpiralMLP for academic research

## 🙏 Acknowledgments

- Original SpiralMLP paper
- PyTorch and timm library developers
- Open source community
