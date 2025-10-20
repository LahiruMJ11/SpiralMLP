# Object Detection with SpiralMLP Backbones

This directory contains the code for running object detection experiments on the COCO dataset. The models leverage backbones like PVT and SpiralMLP within the [MMDetection](https://github.com/open-mmlab/mmdetection) framework.

---

## ⚙️ Setup

### 1. Create Conda Environment

It is highly recommended to use a dedicated Conda environment to manage dependencies.
```bash
# Create a new environment with Python 3.8
conda create -n spiralmlp_det python=3.8 -y

# Activate the environment
conda activate spiralmlp_det
```

### 2. Install Dependencies

Install the required libraries, including the specific versions of PyTorch, MMCV, and MMDetection that this codebase was built on.
```bash
# 1. Install PyTorch with CUDA 11.1
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y

# 2. Install MMCV-full
pip install mmcv-full==1.3.8 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

# 3. Install MMDetection
pip install mmdet==2.13.0

# 4. Install other common packages
pip install timm matplotlib opencv-python-headless
```

---

## 📦 Data Preparation

This project uses the COCO 2017 dataset. You need to download the validation images and annotations and organize them in the following structure inside the `detection` folder:
```
detection/
├── data/
│   └── coco/
│       ├── annotations/
│       │   └── instances_val2017.json
│       └── val2017/
│           ├── 000000000139.jpg
│           ├── 000000000285.jpg
│           └── ...
├── configs/
├── checkpoints/
└── ...
```

**Download Links:**
* [2017 Val images](http://images.cocodataset.org/zips/val2017.zip)
* [2017 Train/Val annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

---

## 🚀 Evaluation

To evaluate a model's performance on the COCO validation set, use the `test.py` script. The command requires a config file and a pretrained checkpoint file.

The general command format for a single GPU is:
```bash
python test.py [CONFIG_FILE] [CHECKPOINT_FILE] --eval bbox
```

---

## 📊 Results

Here are some of the evaluation results achieved with this codebase.

### RetinaNet with PVT-Small Backbone

This test was run using the `retinanet_pvt_s_fpn_1x_coco_640.py` config.

**Command:**
```bash
python test.py configs/retinanet_pvt_s_fpn_1x_coco_640.py checkpoints/retinanet_pvt_s_fpn_1x_coco_640.pth --eval bbox
```

**Output:**
```
[Add your output here]
```

### Mask R-CNN with PVT-v2-B4 Backbone

This test was run using the `mask_rcnn_pvt_v2_b4_fpn_1x_coco.py` config.

**Command:**
```bash
python test.py configs/mask_rcnn_pvt_v2_b4_fpn_1x_coco.py checkpoints/mask_rcnn_pvt_v2_b4_fpn_1x_coco.pth --eval bbox
```

**Output:**
```
[Add your output here]
```

---

## 🚂 Training

To train a model from scratch, you can use the `dist_train.sh` script.

For training on a single GPU (like on a laptop), modify the command to use `1` GPU.
```bash
# Example for training on a single GPU
./dist_train.sh [CONFIG_FILE] 1
```
