# MOGS

This is the repository of MOGS, which has been published on ICRA 2026:

**MOGS: Monocular Object-guided Gaussian Splatting in Large Scenes**

Shengkai Zhang, Yuhe Liu, Jianhua He, Xuedou Xiao, Mozi Chen∗, Kezhong Liu

📌 Abstract
------------------------------------------------------------------------
Recent advances in 3D Gaussian Splatting (3DGS) deliver striking photorealism, and extending it to large scenes opens new opportunities for semantic reasoning and prediction in applications such as autonomous driving. Today’s state-of-theart systems for large scenes primarily originate from LiDARbased pipelines that utilize long-range depth sensing. However, they require costly high-channel sensors whose dense point clouds strain memory and computation, limiting scalability, fleet deployment, and optimization speed. We present MOGS, a monocular 3DGS framework that replaces active LiDAR depth with object-anchored, metrized dense depth derived from sparse visual-inertial (VI) structure-from-motion (SfM) cues. Our key idea is to exploit image semantics to hypothesize per-object shape priors, anchor them with sparse but metrically reliable SfM points, and propagate the resulting metric constraints across each object to produce dense depth. To address two key challenges, i.e., insufficient SfM coverage within objects and cross-object geometric inconsistency, MOGS introduces (1) a multi-scale shape consensus module that adaptively merges small segments into coarse objects best supported by SfM and fits them with parametric shape models. (2) a cross-object depth refinement module that optimizes per-pixel depth under a combinatorial objective combining geometric consistency, prior anchoring, and edge-aware smoothness. Experiments on public datasets show that, with a low-cost VI sensor suite, MOGS reduces training time and memory consumption, while achieving high-quality rendering competitive with costly LiDAR-based approaches in large scenes. 

[![MOGS: Monocular Object-guided Gaussian Splatting in Large Scenes](https://img.youtube.com/vi/6MJs-XAyaKE/hqdefault.jpg)](https://www.youtube.com/watch?v=6MJs-XAyaKE "MOGS: Monocular Object-guided Gaussian Splatting in Large Scenes")
------------------------------------------------------------------------

## 🚀Getting started
### 1. Environment Setup
  This project is tested with the following environment:
  ```
  - Python: 3.10.20
  - PyTorch: 2.5.1 with CUDA 12.1
  - Torchvision: 0.20.1
  - GPU: NVIDIA GPU with CUDA 12.1-compatible driver
  ```
  Please make sure that your NVIDIA driver is compatible with CUDA 12.1.
  ```
  cd MOGS
  conda env create -f MOGS_environment.yml
  conda activate MOGS
  ```
 Most required Python packages and CUDA-related runtime dependencies are already specified in MOGS_environment.yml.

   CUDA runtime libraries and cuDNN are included in the environment dependencies; in most cases, you do not need to manually install an older standalone CUDA 11.0/cuDNN 8.0 stack.

### 2. Clone Repository
  Please align the data based on the timestamps provided by ROS using the following command.
  ```
  git clone --recursive https://github.com/ClarenceZSK/MOGS/.
  ```

### 3. Install Dependencies
  Please download required models.
  ```
  - Depth Anything v3 (DA3)：https://github.com/ByteDance-Seed/Depth-Anything-3
  - SAM2 (Segment Anything 2)：https://github.com/facebookresearch/sam2
  - SuperPoint：https://github.com/rpautrat/SuperPoint
  - LightGlue：https://github.com/cvg/LightGlue
  - GSplat：https://github.com/nerfstudio-project/gsplat
  ```
⚠️ Important: LightGlue, GSplat-main, and project code must be placed under the same directory:

        MOGS/
        ├── .py
        ├── LightGlue
        ├── gsplat

⚠️ If you replace the models, please configure the dependencies accordingly.

## 🚀 Run on Dataset
 

1.Prepare Dataset: Dataset must include ground-truth poses

2.Configure Paths

Modify in:main_rendering.py

    Input image path
    Pose path
    Model paths
    Output directory

3.Run

    cd /MOGS
    python main_rendering.py

The aligned data will be saved in the `mmEMP/data/dataset/synced_data` path.

Pipeline Overview

    Tracking → Triangulation → Depth Estimation → Segmentation→ Clustering → Depth Metricization → 3D Gaussian Splatting

## 📌 Citation
If you find our work useful in your research, please consider citing:
  ```
@InProceedings{Zhang_2026_ICRA,
    author    = {Zhang, Shengkai and Liu, Yuhe and He, Jianhua and Xiao, Xuedou and Chen, Mozi and Liu, Kezhong},
    title     = {MOGS: Monocular Object-guided Gaussian Splatting in Large Scenes},
    booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
    year      = {2026},
}
  ```
⚠️ Notes

    This project relies on multiple external models (DA3, SAM2, etc.)
    Ensure correct model paths and environment configuration
    Performance depends on GPU memory and dataset scale