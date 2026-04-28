MOGS: Monocular Object-guided Gaussian Splatting in Large Scenes
------------------------------------------------------------------------

Abstract

Abstract—Recent advances in 3D Gaussian Splatting (3DGS) deliver striking photorealism, and extending it to large scenes opens new opportunities for semantic reasoning and prediction in applications such as autonomous driving. Today’s state-of-theart systems for large scenes primarily originate from LiDARbased pipelines that utilize long-range depth sensing. However, they require costly high-channel sensors whose dense point clouds strain memory and computation, limiting scalability, fleet deployment, and optimization speed. We present MOGS, a monocular 3DGS framework that replaces active LiDAR depth with object-anchored, metrized dense depth derived from sparse visual-inertial (VI) structure-from-motion (SfM) cues. Our key idea is to exploit image semantics to hypothesize per-object shape priors, anchor them with sparse but metrically reliable SfM points, and propagate the resulting metric constraints across each object to produce dense depth. To address two key challenges, i.e., insufficient SfM coverage within objects and cross-object geometric inconsistency, MOGS introduces 1) a multi-scale shape consensus module that adaptively merges small segments into coarse objects best supported by SfM and fits them with parametric shape models, and 2) a cross-object depth refinement module that optimizes per-pixel depth under a combinatorial objective combining geometric consistency, prior anchoring, and edge-aware smoothness. Experiments on public datasets show that, with a low-cost VI sensor suite, MOGS reduces training time by up to 30.4% and memory consumption by 19.8%, while achieving high-quality rendering competitive with costly LiDAR-based approaches in large scenes. The source code will be publicly available at https://github.com/ClarenceZSK/ MOGS/.


视频
------------------------------------------------------------------------


Installation

1.Begin by cloning this repository and all its submodules using the following command:

    cd MOGS
    git clone --recursive https://github.com/ClarenceZSK/MOGS/.
    
2.Install the dependency model and自己想选或者需要的权重文件.
注意：LightGlue和GSplat-main以及项目代码文件需要放在同级文件夹Run_MOGS下。

    DA3:https://github.com/ByteDance-Seed/Depth-Anything-3
    SA2:https://github.com/facebookresearch/sam2
    Superpoint：https://github.com/rpautrat/SuperPoint
    LightGlue:https://github.com/cvg/LightGlue
    GSplat:https://github.com/nerfstudio-project/gsplat

3.Create an anaconda environment called MOGS.该环境包含所有上述提供链接模型的依赖，如果更换模型，请自己配置环境。

    conda env create -f MOGS_environment.yml
    conda activate MOGS


Run in dataset


1.准备数据集（带位姿真值）

2.根据自己的情况修改项目文件main_rendering.py中的输入输出以及model路径

3.进入项目文件，运行项目

    cd /MOGS/Run_MOGS
    python main_rendering.py

------------------------------------------------------------------------

引用

@article{zhang2025mogs, title={MOGS: Monocular Object-guided Gaussian Splatting in Large Scenes}, author={Zhang, Shengkai and Liu, Yuhe and He, Jianhua and Xiao, Xuedou and Chen, Mozi and Liu, Kezhong},
journal={arXiv preprint arXiv:2509.06685}, year={2025} }

