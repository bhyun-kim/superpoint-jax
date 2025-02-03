# SuperPoint Inference with JAX/Flax

This repository provides an inference-only pipeline for keypoint detection using the SuperPoint model. It includes a PyTorch implementation of SuperPoint, a JAX/Flax (NNX) implementation, and a conversion script to transfer pretrained weights from PyTorch to JAX. Inference can then be performed on input images using the converted JAX model.

## Overview

- **PyTorch Model:** `superpoint_torch.py` implements the SuperPoint model in PyTorch.
- **JAX Model:** `superpoint_jax.py` contains the corresponding SuperPoint model implementation in JAX/Flax NNX.
- **Weight Conversion:** `convert_to_jax.py` includes helper functions to copy convolution and batch normalization parameters from the PyTorch model to the JAX model.
- **Inference Example:** A Jupyter Notebook (`inference.ipynb`) demonstrates the entire process—from converting weights to running inference and visualizing keypoints.

## Requirements

- Python 3.10+
- [PyTorch](https://pytorch.org/)
- [JAX](https://github.com/google/jax) and [jaxlib](https://github.com/google/jax)
- [Flax](https://flax.readthedocs.io/)
- NumPy
- Matplotlib
- OpenCV (cv2)

## Repository Structure

- `superpoint_torch.py`  
  PyTorch implementation of the SuperPoint model.

- `superpoint_jax.py`  
  JAX/Flax (NNX) implementation of the SuperPoint model.

- `convert_to_jax.py`  
  Weight conversion utilities for copying parameters from the PyTorch model to the JAX model.

- `inference.ipynb`  
  Jupyter Notebook demonstrating weight conversion, inference, and side-by-side visualization of keypoint detection.

- `README.md`  
  This file.

## Usage

1. **Obtain Pretrained Weights:**  
   Place the pretrained PyTorch SuperPoint weights (e.g., `superpoint_torch_weights.pth`) in the repository root or a designated folder.

2. **Convert Weights:**  
   Run the conversion script to create a JAX model with copied weights and save the converted state:
   ```bash
   python convert_superpoint_model.py
