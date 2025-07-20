# TreeCoG: Contour Graph Network for Instance Tree Segmentation

## Overview

**TreeCoG** is a novel approach for instance segmentation of individual tree crowns from UAV-captured RGB images, designed for biodiversity monitoring in dense tropical forests. The method leverages contour-based over-segmentation and graph-based learning to achieve precise and efficient instance segmentation. Our method outperforms state-of-the-art models in both accuracy and speed.

This repository includes the implementation of the **TreeCoG** framework along with the new benchmark dataset **ForestSeg**, which is collected over multiple seasons in Vietnamese tropical forests.

## Features

-  **Contour-based Segmentation**: Deliberate over-segmentation using a deep edge detector to reduce instance boundary ambiguity.
-  **Graph Neural Network**: Learns to merge contours into full tree instances by modeling their spatial and visual relationships.
- **High Accuracy**: Achieves **57.01% AP**, **62.21% AP@50**, and **55.32% AP@70** on ForestSeg benchmark.
- **Efficient & Lightweight**: Uses only **10.81M parameters** and infers at **6.2 ms/image**—ideal for real-time UAV deployment.
- 🌲**Robust Dataset**: The ForestSeg dataset includes seasonal and altitudinal variations from 4 UAV flight campaigns.

## Installation

### Prerequisites

- Python ≥ 3.8
- PyTorch ≥ 1.9
- CUDA-enabled GPU (optional but recommended)
- pip (Python package installer)

### Setup

```bash
# Clone the repository
git clone https://github.com/DoMaiVietHoang/Contour-Graph-for-Image-instance-segmentation.git
cd Contour-Graph-for-Image-instance-segmentation

# Install dependencies
pip install -r requirements.txt
```


## Model Architecture

TreeCoG comprises three main stages:

1. **Contour Extraction**:
   - Uses **EDTER** (Edge Detection Transformer) with BiMLA decoder to generate fine-grained contour maps.
   - Applies Gaussian blur + Guo-Hall thinning for clean edge representation.

2. **Contour Feature Representation**:
   - Constructs a graph where each node is a contour.
   - Computes:
     - **Appearance features** using LPIPS (Learned Perceptual Image Patch Similarity)
     - **Shape features**: area, extent, solidity, aspect ratio, deviation

3. **Graph-based Contour Merging**:
   - A Graph Convolutional Network (GCN) is trained to predict which contours should be merged into the same instance.
   - Outputs instance masks after graph-based edge classification.

## ForestSeg Dataset

- UAV-based RGB dataset collected in tropical Vietnamese forests.
- Contains **4 subsets** across different dates and altitudes (ForestSeg-T1 to T4).
- Publicly available at: [https://sigm-seee.github.io/datasets/ForestSeg.html](https://sigm-seee.github.io/datasets/ForestSeg.html)

| Subset        | #Train | #Test | Altitude | Drone         |
|---------------|--------|-------|----------|----------------|
| ForestSeg-T1  | 1344   | 480   | 70m      | DJI Phantom 4 RTK |
| ForestSeg-T2  | -      | 410   | 211m     | DJI Air 3         |
| ForestSeg-T3  | -      | 350   | 150m     | DJI Air 3         |
| ForestSeg-T4  | -      | 360   | 100m     | DJI Air 3 (3x Zoom) |

## Benchmark Performance

| Method               | Params (M) | Time (ms) | AP   | AP@50 | AP@70 |
|----------------------|------------|-----------|------|--------|--------|
| Mask R-CNN (ResNet50)| 43.05      | 8.3       | 30.63| 46.23  | 26.17  |
| Mask R-CNN (Swin-T)  | 48.55      | 10.8      | 56.72| 60.12  | 54.64  |
| YOLOv11              | 22.33      | 7.5       | 38.30| 52.78  | 33.51  |
| **TreeCoG (Ours)**   | **10.81**  | **6.2**   | **57.01**| **62.21**| **55.32** |



