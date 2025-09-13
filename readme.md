# EC-UNIT
This repository provides the official PyTorch implementation for the following paper:
**Enhancing Cross-domain Correspondence for Unsupervised Image-to-Image Translation**
> **Abstract:** *UNsupervised Image-to-image Translation (UNIT) aims to translate images across visual domains without paired training data, which has been widely used in style transfer, image processing, game design, etc. However, ensuring the correspondence (e.g., target category, pose, or head orientation) between generated images and inputs remains a formidable challenge. To this end, we present a new scheme, named EC-UNIT, which comprises three innovative designs aiming to Enhance cross-domain Correspondence for UNIT. Specifically, 1) we propose Multi-level Style Embedding to extract multi-level style features for fusion while imposing our newly designed Hierarchical Consistency Constraints on both the content and style features (MSE\&HCC), aiming to retain more style representations and facilitate feature disentanglement; 2) we develop Semantic Perceptual Matching (SPM) to minimize the semantic distribution discrepancy between the generated image and the input image by leveraging the multimodal model CLIP, dedicated to enhancing semantic consistency;  3) considering that previous works have struggled to control the image translation using pixel-level visual consistency constraints, we design Visual Perceptual Guidance (VPG) to reduce the perceptual distance between the generated image and the style input in VGG feature space, devoted to enhancing visual perceptual correspondence, thereby preventing the generation of unrealistic image details. Extensive experiments demonstrate that our EC-UNIT is more stable and outperforms current SOTA competitors in terms of image quality and diversity as well as both content and style consistency.*

## Installation
```bash
git clone https://github.com/GZHU-DVL/EC-UNIT.git
cd EC-UNIT
```
**Dependencies:**

We have tested on:
- CUDA 12.1
- PyTorch 2.1.2

All dependencies for defining the environment are provided in `environment/EC-UNIT.yaml`.

```bash
conda env create -f ./environment/EC-UNIT.yaml
```

## Installing CLIP into the Environment
Download the CLIP source code as a ZIP archive from [Github](https://github.com/openai/CLIP). After downloading, unzip the file, navigate into the extracted directory, and install the required environment before starting training.

```bash
conda activate EC-UNIT
wget https://github.com/openai/CLIP/archive/refs/heads/main.zip -O CLIP-main.zip
cd CLIP-main/
pip install -e .
```

## Dataset Preparation
| Translation Task | Used Dataset                                                                                                                                                                                                                                                                           | 
|:-----------------|:-----------------| 
| Male←→Female     | [CelebA-HQ](https://github.com/clovaai/stargan-v2#datasets-and-pre-trained-networks) ( divided into male and female subsets by [StarGANv2](https://github.com/clovaai/stargan-v2#datasets-and-pre-trained-networks))                                                                     |
| Dog←→Cat         | [AFHQ](https://github.com/clovaai/stargan-v2#datasets-and-pre-trained-networks) ( provided by [StarGANv2](https://github.com/clovaai/stargan-v2#datasets-and-pre-trained-networks))                                                                                                       |
| Face←→Cat        | [CelebA-HQ](https://github.com/switchablenorms/CelebAMask-HQ) and [AFHQ](https://github.com/clovaai/stargan-v2#datasets-and-pre-trained-networks)                                                                                                                                      |
| Bird←→Dog        | 4 classes of birds and 4 classes of dogs in [ImageNet291](https://github.com/williamyang1991/GP-UNIT/tree/main/data_preparation)
| Bird←→Car        | 4 classes of birds and 4 classes of cars in [ImageNet291](https://github.com/williamyang1991/GP-UNIT/tree/main/data_preparation)                                                                
## Pretrained Models
Pretrained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/10GGr3LsNeS7AZV7GKXeGzm1Us3Md20w9?usp=drive_link). 
<!-- (The Content Encoder should be downloaded and placed in `./checkpoint/content_encoder.pt` before main training.) -->
                             
## Image-to-Image Translation Training
```python
python train.py --task TASK --batch BATCH_SIZE --iter ITERATIONS \
                --source_paths SPATH1 SPATH2 ... SPATHS --source_num SNUM1 SNUM2 ... SNUMS \
                --target_paths TPATH1 TPATH2 ... TPATHT --target_num TNUM1 TNUM2 ... TNUMT
```
### Instance
__Cat->Dog__

```python
python  train.py  --task cat2dog_EC-UNIT \
    --source_paths ../../dataset/afhq/train/cat  \
    --source_num 4000 \
    --target_paths ../../dataset/afhq/train/dog \
    --target_num 4000 \
    --mitigate_style_bias \
    --batch 10  \
    --target_type dog \
    --iter 180000
```
## Generate Images for Visualization
### Instance
```python
python gene_for_viz_or_metrics.py \
--content_encoder_path ./checkpoint/content_encoder.pt \
--generator_path ./checkpoint/cat2dog_EC-UNIT.pt  \
--mode viz \
--target_paths ../../dataset/afhq/val/dog/ \
--source_paths ../../dataset/afhq/val/cat/ \
--ref_img_paths ../../dataset/afhq/train/dog/ \
--trg_num 500 \
--src_num 500 \
--src_domain cat \
--trg_domain dog  \
--device cuda \
--task generate_images_for_visualization

```
## Generate Images for metrics
### Instance
```python
python gene_for_viz_or_metrics.py \
--content_encoder_path ./checkpoint/content_encoder.pt \
--generator_path ./checkpoint/cat2dog_EC-UNIT.pt  \
--mode calc \
--target_paths ../../dataset/afhq/val/dog/ \
--source_paths ../../dataset/afhq/val/cat/ \
--ref_img_paths ../../dataset/afhq/train/dog/ \
--trg_num 500 \
--src_num 500 \
--src_domain cat \
--trg_domain dog  \
--device cuda \
--task generate_images_for_metrics
```
## Acknowledgments

The code is developed based on GP-UNIT: https://github.com/williamyang1991/GP-UNIT