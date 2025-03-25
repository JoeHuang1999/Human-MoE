# Human-MoE: Multimodal Full-Body Human Image Synthesis with Component-driven Mixture of Experts
Yu-Jiu Huang and I-Chen Lin
## Description
This repository provides the PyTorch implementation of the paper "Human-MoE: Multimodal Full-Body Human Image Synthesis with Component-driven Mixture of Experts".
## Requirements
- Ubuntu 20.04 or Windows 11
- CUDA version 11.6 or later
- [Anaconda](https://www.anaconda.com/download)
### Installation
1. Create and activate the virtual environment.
```
conda create -n ldm python=3.9
conda activate ldm
```
2. Install the packages.
```
cd Human-MoE
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
## Datasets
### [DeepFashion-MultiModal](https://github.com/yumingj/DeepFashion-MultiModal)
We use the dataset preprocessed by [Text2Human](https://github.com/yumingj/Text2Human) as the starting point.

The background from the ground truth images will be removed and placed into ```./data/no-bg-train-images``` and ```./data/no-bg-test-images```, respectively.

Parsing maps are stored in ```./data/masks```, pose maps are extracted using OpenPose and saved in ```./data/skeletons```, and captions are obtained via the Gemini API and placed in ```./data/captions```.


```
./data
├── no-bg-train-images
    ├── *.png
├── no-bg-test-images
    ├── *.png
├── masks
    ├── *.png
├── skeletons
    ├── *.png
├── captions
    ├── *.txt
├── binary-masks
    ├── 1
        ├── *.png
    ├── 2
    ...
    └── 25
├── face-refinement
    ├── no-bg-train-images
    ├── no-bg-test-images
    ├── masks
    ├── skeletons
    └── captions
├── hand-refinement
├── upper-garment-refinement
└── lower-garment-refinement
```
## Coming Soon
- 
