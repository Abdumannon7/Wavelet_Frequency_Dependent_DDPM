# Wavelet Frequency-Dependent DDPM

A Diffusion Probabilistic Model (DDPM) for medical image generation that leverages wavelet transforms and frequency-dependent noise scheduling. This project uses the BraTS dataset to generate synthetic medical brain MRI images.

## Project Overview

This codebase implements a frequency-adaptive denoising diffusion probabilistic model that:
- Decomposes images into frequency bands using discrete wavelet transforms (DWT)
- Uses adaptive frequency-dependent training for different wavelet bands
- Generates high-quality synthetic medical images for data augmentation
- Includes a binary tumor classifier for evaluating generated images

## File Structure

### Core Model Components
- **`ddpm.py`** - Denoising Diffusion Probabilistic Model implementation with noise scheduling
- **`unet.py`** - U-Net architecture used as the backbone denoiser for diffusion model
- **`dwt_idwt_transforms.py`** - Discrete Wavelet Transform (DWT) and Inverse DWT utilities for frequency decomposition

### Training & Sampling
- **`train_model.py`** - Main training script for the diffusion model; configure via `configuration.yml`
- **`sample_model.py`** - Generate new images using trained diffusion models with metric evaluation
- **`bulk_generate.py`** - Batch generation of synthetic images for augmentation

### Utilities
- **`datafilters.py`** - BraTS dataset loading, filtering, and preprocessing utilities
- **`decode.py`** - HDF5 file decoding for BraTS medical image data
- **`wavelet_energy_analysis.py`** - Wavelet energy analysis and frequency band visualization

### Classification & Evaluation
- **`train_classifier.py`** - Train ResNet18 binary classifier to distinguish real vs generated images
- **`evaluate_classifier.py`** - Evaluate classifier performance and generate metrics

### Configuration
- **`configuration.yml`** - Central configuration file for all hyperparameters

## How to Run

### Prerequisites
1. Create a Python environment

2. Download and prepare BraTS2020 dataset:
   - Place data at the path specified in `configuration.yml` under `dataset_params.data_root`
   - Update CSV path in `configuration.yml` to point to your BraTS metadata CSV

3. Update `configuration.yml` with your paths and hyperparameters

### Training the Diffusion Model
```bash
python train_model.py --config_path configuration.yml
```
This trains frequency-dependent UNet models for different wavelet bands and saves checkpoints to the output folder.

### Generating Images
```bash
python sample_model.py --config_path configuration.yml --checkpoint_path <path_to_model.pth>
```
Generates new synthetic images and computes quality metrics (SSIM, FID).

### Bulk Generation (Augmentation)
```bash
python bulk_generate.py --config_path configuration.yml --checkpoint_path <path_to_model.pth> --num_samples <N>
```
Generates N synthetic images in batch and saves them for dataset augmentation.

### Training the Classifier
```bash
python train_classifier.py --config_path configuration.yml
```
Trains a ResNet18 classifier on real vs generated images to evaluate model quality.

### Evaluating Classifier
```bash
python evaluate_classifier.py --config_path configuration.yml --checkpoint_path <path_to_classifier.pth>
```
Evaluates classifier performance on test set and reports metrics.

