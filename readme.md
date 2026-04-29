MOGS: Monocular Object-guided Gaussian Splatting in Large Scenes

📌 Official implementation of MOGS
🔗 Project Page: https://github.com/ClarenceZSK/MOGS/
------------------------------------------------------------------------

Abstract

Abstract—Recent advances in 3D Gaussian Splatting (3DGS) deliver striking photorealism, and extending it to large scenes opens new opportunities for semantic reasoning and prediction in applications such as autonomous driving. Today’s state-of-theart systems for large scenes primarily originate from LiDARbased pipelines that utilize long-range depth sensing. However, they require costly high-channel sensors whose dense point clouds strain memory and computation, limiting scalability, fleet deployment, and optimization speed. We present MOGS, a monocular 3DGS framework that replaces active LiDAR depth with object-anchored, metrized dense depth derived from sparse visual-inertial (VI) structure-from-motion (SfM) cues. Our key idea is to exploit image semantics to hypothesize per-object shape priors, anchor them with sparse but metrically reliable SfM points, and propagate the resulting metric constraints across each object to produce dense depth. To address two key challenges, i.e., insufficient SfM coverage within objects and cross-object geometric inconsistency, MOGS introduces 1) a multi-scale shape consensus module that adaptively merges small segments into coarse objects best supported by SfM and fits them with parametric shape models, and 2) a cross-object depth refinement module that optimizes per-pixel depth under a combinatorial objective combining geometric consistency, prior anchoring, and edge-aware smoothness. Experiments on public datasets show that, with a low-cost VI sensor suite, MOGS reduces training time by up to 30.4% and memory consumption by 19.8%, while achieving high-quality rendering competitive with costly LiDAR-based approaches in large scenes. The source code will be publicly available at https://github.com/ClarenceZSK/MOGS/.


🎥 Video
------------------------------------------------------------------------

🚀 Installation

1.Clone Repository

    cd MOGS
    git clone --recursive https://github.com/ClarenceZSK/MOGS/.

2.Install Dependencies：Download required models

    Depth Anything v3 (DA3)：https://github.com/ByteDance-Seed/Depth-Anything-3
    SAM2 (Segment Anything 2)：https://github.com/facebookresearch/sam2
    SuperPoint：https://github.com/rpautrat/SuperPoint
    LightGlue：https://github.com/cvg/LightGlue
    GSplat：https://github.com/nerfstudio-project/gsplat

⚠️ Important:LightGlue, GSplat-main, and project code must be placed under the same directory:

        MOGS/
        ├── .py
        ├── LightGlue
        ├── gsplat


3.Environment Setup

    conda env create -f MOGS_environment.yml
    conda activate MOGS

⚠️ If you replace the models, please configure the dependencies accordingly.


🚀 Run on Dataset

1.Prepare Dataset:Dataset must include ground-truth poses

2.Configure Paths

Modify in:main_rendering.py

    Input image path
    Pose path
    Model paths
    Output directory

3.Run

    cd /MOGS
    python main_rendering.py

Pipeline Overview

    Tracking → Triangulation → Depth Estimation → Segmentation→ Clustering → Depth Metricization → 3D Gaussian Splatting

📌 Citation

If you find this work useful, please cite:

@article{zhang2025mogs,
  title={MOGS: Monocular Object-guided Gaussian Splatting in Large Scenes},
  author={Zhang, Shengkai and Liu, Yuhe and He, Jianhua and Xiao, Xuedou and Chen, Mozi and Liu, Kezhong},
  journal={arXiv preprint arXiv:2509.06685},
  year={2025}
}

⚠️ Notes

    This project relies on multiple external models (DA3, SAM2, etc.)
    Ensure correct model paths and environment configuration
    Performance depends on GPU memory and dataset scale