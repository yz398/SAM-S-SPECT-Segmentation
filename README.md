# SAM-S: SAM-Based Tumor Segmentation in SPECT Imaging

This repository provides the Python implementation of **SAM-S**, a Segment Anything Model (SAM)-based approach for individual tumor segmentation in single-photon emission computed tomography (SPECT) imaging.

The code accompanies our research comparing AI-based segmentation approaches with conventional thresholding methods for individual tumor segmentation in radiopharmaceutical therapy (RPT).

The objective is to investigate the feasibility of applying foundation models to SPECT tumor segmentation and evaluate segmentation performance in the context of absorbed-dose estimation.

## 1. Overview

SAM-S is developed using the pretrained Segment Anything Model (SAM), adapted for tumor segmentation in SPECT images.

The framework includes:

- SPECT image preprocessing and dataset preparation
- Prompt-based segmentation using bounding boxes and/or points
- Fine-tuning of the SAM mask decoder
- Patient-level segmentation inference
- Reconstruction of 3D segmentation volumes from 2D predictions

The implementation supports three prompting strategies:

- **Point prompts**
- **Bounding-box prompts**
- **Combined point and bounding-box prompts**

**Note:** In the current implementation, prompts are generated from reference segmentation masks. Therefore, the code represents reference-guided segmentation rather than a fully automatic tumor detection and segmentation pipeline.

## 2. Model Architecture

The implementation uses the pretrained SAM ViT-Base model available through Hugging Face:

[facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base)

During fine-tuning:

- The pretrained image encoder is frozen.
- The prompt encoder is frozen.
- Only the mask decoder is updated.
- A combined Dice and cross-entropy loss is used for optimization.

This strategy allows the pretrained model to be adapted to SPECT images while reducing the number of trainable parameters.

### Bounding-Box Prompts

Bounding boxes are generated from reference tumor masks and enlarged by a fixed number of pixels.

The default enlargement is **5 pixels in each direction**, subject to image boundaries.

### Segmentation Workflow

1. Load SPECT images and reference tumor segmentations.
2. Normalize and prepare the images for SAM.
3. Generate prompts from the reference masks.
4. Apply the fine-tuned SAM model to individual 2D slices.
5. Convert predicted probabilities to binary segmentation masks.
6. Combine predictions from the same physical slice.
7. Reconstruct patient-level segmentation volumes.
