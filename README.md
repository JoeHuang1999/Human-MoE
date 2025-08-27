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

The background from the ground truth images will be removed and placed into ```./srcdata/no-bg-train-images``` and ```./srcdata/no-bg-test-images```, respectively.

Parsing maps are stored in ```./src/data/masks```, pose maps are extracted using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and saved in ```./src/data/skeletons```, and captions are obtained via the [BLIP-2](https://github.com/salesforce/LAVIS) (for full-body images) or [Gemini API](https://ai.google.dev/) (for others), then placed in ```./src/data/captions```.

Binary masks are created based on parsing labels or randomly generated strokes and placed in ```./src/data/binary-masks```. The dataset contains 23 categories (excluding the background). For each category, a binary mask is generated based on its corresponding category values. These masks are stored in directories ranging from ```./src/data/binary-masks/1``` to ```./src/data/binary-masks/23```. The "upper garment" category is stored in ```./src/data/binary-masks/24```, while the "lower garment" category is stored in ```./src/data/binary-masks/25```. Additionally, the script ```./src/tools/random_mask_generator.py``` generates random binary masks, which are stored in ```./src/data/binary-masks/26```.

For the experts dataset, ground truth images undergo the following processing: A fine-tuned [YOLO V9](https://github.com/WongKinYiu/yolov9) model detects face and hand BBOXes, cropping them to 128×128, while upper and lower clothing BBOXes are extracted from the parsing map and resized to 256×256. We perform similar operations on the ground truth as those applied to full-body images to obtain the corresponding conditions.

The final folder structure is as follows.
```
./src/data
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
    └── 26
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
## Pre-trained Weights
We provide pre-trained weights for convenience. You can download them from [Google Drive](https://drive.google.com/drive/folders/1VlOBth8SlnolHoqwcWIwK0cxwiNlSKkY?usp=sharing). After downloading, please place the weights into ```./src```

## Inference
We provide a simple GUI for running inference with the pre-trained model.  

### Running the Demo
Execute the following command to launch the GUI and generate results.
```
python ui_demo.py
```
After running the command, you will see the following user interface.

<img width="1600" height="900" alt="使用者介面完整版" src="https://github.com/user-attachments/assets/b4a5e2ea-6ea4-4093-b1b7-a8d7bbb688f9" />

The output will be saved at ```./HumanMoE/src/deepfashion/cond_text_image_samples/result.png```
You can further enhance the generated results by sequentially running the refinement commands.
```
python ui_refine_face.py
python ui_refine_hand.py
python ui_refine_upper_garment.py
python ui_refine_lower_garment.py
```
### Batch Sampling
#### Full-body image generation
```
python ldm_sample_folder.py
```
#### Part-specific image generation
```
python ldm_sample_folder_face.py
python ldm_sample_folder_hand.py
python ldm_sample_folder_lower_garment.py
python ldm_sample_folder_upper_garment.py
```
#### Full-body image inpainting
```
python ldm_sample_folder_inpainting_nn.py
```

## Training
The training process can be divided into two major steps: **training the autoencoder** and **training the LDM**.
First, navigate to the ```./src/tools``` directory.
### 1. Autoencoder (VQ-VAE)
```
python train_vqvae.py
```
### 2. LDM
```
python train_ddpm_cond_deepfashion.py
```
These commands demonstrate the standard full-body training pipelines for both the autoencoder and the LDM. If you need to train models for other regions (e.g., face, hand, upper garment, lower garment), simply modify the dataset class and configuration file inside ```train_vqvae.py``` or ```train_ddpm_cond_deepfashion.py```. The training procedure and logic remain the same; only the dataset and configuration differ.








