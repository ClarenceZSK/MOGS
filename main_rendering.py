"""
主流程：逐帧交错处理
流程：Tracking → Triangulation → CSV输出 → DA3+SA2 → Cluster → Supervision → GSplat
"""

import os
import sys
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LC_ALL"] = "C.UTF-8"
os.environ["LANG"] = "C.UTF-8"

try:
    import io
    import torch.utils._cpp_extension_versioner as _ext_ver
    def _hash_source_files_utf8(hash_value, source_files):
        for src in source_files:
            try:
                with io.open(src, encoding="utf-8", errors="ignore") as f:
                    hash_value = _ext_ver.update_hash(hash_value, f.read())
            except Exception:
                pass
        return hash_value
    _ext_ver.hash_source_files = _hash_source_files_utf8
except Exception:
    pass

import gc
import csv
import tempfile
import shutil
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import cv2
import pandas as pd

from gsplat import rasterization

# 自定义模块
import tracking
import triangulation
import model
import cluster
import supervision as sv
from gaussian_splatting import StreamingGaussianSplatting, GaussianParams, CameraParams, StrategyParams


# ===================== 路径与参数配置 =====================
IMAGE_FOLDER = "/data/WYM/kitti/odometry/dataset/sequences/00/image0_900"
POSE_PATH = "/data/WYM/kitti/odometry/poses/00.txt"
SUPERPOINT_WEIGHTS = "/data/WYM/models/SL/superpoint_v1.pth"

DA3_MODEL_DIR = "/data/WYM/models/DA3/giant_model_json"
SA2_CHECKPOINT = "/data/WYM/models/SA2/checkpoints/sam2.1_hiera_base_plus.pt"
SA2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# [OUTPUT DISABLED] 中间模块输出路径（可视化/调试图像已禁用，若有需要可修改为自己的路径）
OUTPUT_ROOT_DIR = "/data/WYM/models/SL/run/output/point_images"
VIS_SAVE_DIR = "/data/WYM/models/SL/run/output/vis_frames"
DA3_OUTPUT_FOLDER = "/data/WYM/models/SL/run/output/da3_depth"
SA2_OUTPUT_BASE = "/data/WYM/models/SL/run/output/sa2_mask"
SA2_MASK_FOLDER = os.path.join(SA2_OUTPUT_BASE, "masks")
SA2_VIS_FOLDER = os.path.join(SA2_OUTPUT_BASE, "vis")
CLUSTER_MASK_DIR = "/data/WYM/models/SL/run/output/cluster/bitmaps_merged"
CLUSTER_CSV_DIR  = "/data/WYM/models/SL/run/output/cluster/per_frame_csvs"
CLUSTER_VIS_DIR = "/data/WYM/models/SL/run/output/cluster/mask_image"
CLUSTER_POINT_VIS_DIR = "/data/WYM/models/SL/run/output/cluster/mask_point"
SUPERVISION_OUTPUT_DIR = "/data/WYM/models/SL/run/output/supervision"
SUPERVISION_MASK_PATTERN = os.path.join(CLUSTER_MASK_DIR, "{frame:06d}_bitmaps.npz")
GS_OUTPUT_DIR = "/data/WYM/models/SL/run/output/gsplat"

START_FRAME = 0
END_FRAME = 900
MAX_KEYPOINTS = 1024
ENABLE_SA2_VISUALIZATION = True


# ===================== Supervision 参数 ===================== 
SV_D_MAX = 100.0
SV_TAU_SKY = 0.01
SV_PLANE_USE = True
SV_PLANE_ITERS = 200
SV_PLANE_THRESH = 0.10
SV_PLANE_IMPROVE_RATIO = 0.15


# ===================== SA2 调参区域 =====================
SA2_POINTS_PER_SIDE = 40
SA2_PRED_IOU_THRESH = 0.80
SA2_STABILITY_SCORE_THRESH = 0.80
SA2_BOX_NMS_THRESH = 0.40
SA2_CROP_N_LAYERS = 0
SA2_CROP_N_POINTS_DOWNSCALE_FACTOR = 0
SA2_MIN_MASK_REGION_AREA = 0


# ===================== GSplat 参数 =====================
@dataclass
class FilterParams:    #每帧初始化前，对上一帧高斯进行筛选的参数。
    max_remove_ratio: float = 0.70       # 最多删除比例
    opa_threshold: float = 0.01          # 不透明度阈值
    color_error_threshold: float = 0.0   # 颜色误差阈值                                 
FILTER_CONFIG = FilterParams()


# ===================== GSplat 核心训练参数=====================
GS_ITERS = 1000
GS_EARLY_STOP_LOSS = 0.001
GS_CAMERA = CameraParams(
    width=1241,
    height=376,
    fx=718.856,
    fy=718.856,
    cx=607.1928,
    cy=185.2157,
)

GS_CONFIG = GaussianParams(
    sh_degree=3,
    init_opacity=0.1,
    init_scale=0.001,
    scale_min=0.001,
    scale_max=0.1,
    lr_means=1.6e-4,
    lr_quats=1.6e-3,
    lr_scales=5e-3,
    lr_opacities=8e-3,
    lr_sh=8e-3,                        
    loss_l1_weight=0.8,
    loss_ssim_weight=0.2,
    lr_decay_rate=0.99,
    lr_decay_interval=100,
    scale_clamp_min=0.001,
    scale_clamp_max=0.5,
    near_plane=0.001,
    far_plane=1500.0,
)

GS_STRATEGY = StrategyParams(
    verbose=False,
    prune_opa=0.005,
    grow_grad2d=0.0002,
    grow_scale3d=0.05,
    max_gaussians=150000,
    split_scale_factor=1.6,
    densify_interval=1000,
)


# ===================== 环境设置 =====================
os.environ["OPENCV_OPENCL_RUNTIME"] = ""
os.environ["QT_QPA_PLATFORM"] = "offscreen"
TEMP_DIR = "/data/WYM/tmp"
os.makedirs(TEMP_DIR, exist_ok=True)
tempfile.tempdir = TEMP_DIR
os.environ["TMPDIR"] = TEMP_DIR
os.environ["TEMP"] = TEMP_DIR
os.environ["TMP"] = TEMP_DIR
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"


# ===================== 辅助函数 =====================
def ensure_rgb_uint8(rendered_tensor, target_shape=None):
    if isinstance(rendered_tensor, torch.Tensor):
        rendered = rendered_tensor.detach().cpu().numpy()
    else:
        rendered = np.array(rendered_tensor)
    
    if rendered.ndim == 3:
        if rendered.shape[0] == 3 and rendered.shape[1] > 3 and rendered.shape[2] > 3:
            rendered = np.transpose(rendered, (1, 2, 0))
        elif rendered.shape[-1] == 1:
            rendered = np.repeat(rendered, 3, axis=-1)
        elif rendered.shape[-1] == 3:
            pass
        else:
            rendered = rendered[..., :3]
    elif rendered.ndim == 2:
        print(f"  [GS] 警告：渲染输出为2D，转为3通道灰度")
        rendered = np.stack([rendered] * 3, axis=-1)
    else:
        raise ValueError(f"无法处理的渲染图维度: {rendered.shape}")
    
    if rendered.max() > 1.5:
        rendered = np.clip(rendered, 0, 255) / 255.0
    else:
        rendered = np.clip(rendered, 0, 1)
    rendered = (rendered * 255).astype(np.uint8)

    if target_shape is not None and rendered.shape[:2] != target_shape[:2]:
        rendered = cv2.resize(rendered, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    
    return rendered


def evaluate_gaussian_color_errors(state, image_rgb, pose_w2c, camera_config, device):
    device = torch.device(device)
    means = state['means'].to(device)
    quats = state['quats'].to(device)
    scales = state['scales'].to(device)
    opacities = torch.sigmoid(state['opacities'].to(device))
    colors = torch.sigmoid(state['colors'].to(device))
    
    viewmat = torch.from_numpy(pose_w2c).float().to(device)
    K = camera_config.get_intrinsics_matrix(device)
    H, W = camera_config.height, camera_config.width
    
    image_gt = torch.from_numpy(image_rgb).float().to(device) / 255.0
    
    with torch.no_grad():
        rendered, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales.clamp(GS_CONFIG.scale_clamp_min, GS_CONFIG.scale_clamp_max),
            opacities=opacities,
            colors=colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=GS_CONFIG.sh_degree,
            backgrounds=torch.zeros(3, device=device)[None, :],
            near_plane=GS_CONFIG.near_plane,
            far_plane=GS_CONFIG.far_plane,
            render_mode='RGB',
            packed=False,
        )
        rendered = rendered[0]
    
    ones = torch.ones(len(means), 1, device=device)
    means_h = torch.cat([means, ones], dim=-1)
    means_cam = (viewmat @ means_h.T).T[:, :3]
    
    valid_depth = means_cam[:, 2] > GS_CONFIG.near_plane
    means_proj = (K @ means_cam.T).T
    z = means_proj[:, 2] + 1e-6
    u = means_proj[:, 0] / z
    v = means_proj[:, 1] / z
    
    in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H) & valid_depth
    
    errors = torch.zeros(len(means), device=device)
    if in_image.any():
        u_int = torch.clamp(u.long(), 0, W - 1)
        v_int = torch.clamp(v.long(), 0, H - 1)
        vis_idx = torch.where(in_image)[0]
        err = torch.abs(rendered[v_int[vis_idx], u_int[vis_idx]] - image_gt[v_int[vis_idx], u_int[vis_idx]]).mean(dim=-1)
        errors[vis_idx] = err
    
    return errors.cpu()


def filter_gaussians(state, color_errors, config: FilterParams):
    N = len(state['opacities'])
    max_remove = int(N * config.max_remove_ratio)
    
    opacities = torch.sigmoid(state['opacities'])
    low_opa_mask = opacities < config.opa_threshold
    
    n_must_remove = low_opa_mask.sum().item()
    remaining_budget = max(0, max_remove - n_must_remove)
    
    candidate_mask = (~low_opa_mask) & (color_errors > config.color_error_threshold)
    high_error_mask = torch.zeros(N, dtype=torch.bool)
    
    if remaining_budget > 0 and candidate_mask.any():
        candidate_errors = color_errors.clone()
        candidate_errors[~candidate_mask] = -1.0
        k = min(remaining_budget, candidate_mask.sum().item())
        _, topk_idx = torch.topk(candidate_errors, k=k)
        high_error_mask[topk_idx] = True
    
    delete_mask = low_opa_mask | high_error_mask
    keep_mask = ~delete_mask
    
    actual_remove = delete_mask.sum().item()    
    return keep_mask


def extract_new_points_from_depth(image, depth, pose_w2c, camera_config, device):
    device = torch.device(device)
    H, W = image.shape[:2]
    image_t = torch.from_numpy(image).float().to(device) / 255.0
    depth_t = torch.from_numpy(depth).float().to(device)
    
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = torch.from_numpy(u).float().to(device)
    v = torch.from_numpy(v).float().to(device)
    
    Z = depth_t
    X = (u - camera_config.cx) * Z / camera_config.fx
    Y = (v - camera_config.cy) * Z / camera_config.fy
    points_cam = torch.stack([X, Y, Z], dim=-1)
    
    # 世界坐标系转换
    R_w2c = torch.from_numpy(pose_w2c[:3, :3]).float().to(device)
    t_w2c = torch.from_numpy(pose_w2c[:3, 3]).float().to(device)
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c
    
    valid_mask = (Z > 0.1) & (Z < 100.0) & torch.isfinite(Z)
    valid_points_cam = points_cam[valid_mask]
    valid_colors = image_t[valid_mask]
    
    valid_points_world = (R_c2w @ valid_points_cam.T).T + t_c2w
    
    target_num = min(100000, len(valid_points_world))
    if len(valid_points_world) > target_num:
        indices = torch.randperm(len(valid_points_world))[:target_num]
        valid_points_world = valid_points_world[indices]
        valid_colors = valid_colors[indices]
    
    return valid_points_world.cpu().numpy(), valid_colors.cpu().numpy()


def build_gaussian_params_from_points(points, colors, sh_dim, device):
    device = torch.device(device)
    n = len(points)
    if n == 0:
        return None
    
    means = torch.from_numpy(points).float().to(device)
    colors_t = torch.from_numpy(colors).float().to(device)
    
    colors_full = torch.zeros(n, sh_dim, 3, device=device)
    colors_full[:, 0, :] = colors_t
    
    quats = StreamingGaussianSplatting._random_quats(n, device=device)
    scales = torch.ones(n, 3, device=device) * 0.01
    opacities = torch.ones(n, device=device) * 0.1
    
    return means, quats, scales, opacities, colors_full


# ===================== 主函数 =====================
def main():
    # ==================== 输出目录创建（目前仅保留 GS 渲染输出）====================
    # [OUTPUT DISABLED] 中间结果目录创建已注释（子模块内部makedirs）
    # Path(OUTPUT_ROOT_DIR).mkdir(parents=True, exist_ok=True)
    frame_csv_dir = Path(OUTPUT_ROOT_DIR) / "per_frame_csvs"
    # frame_csv_dir.mkdir(exist_ok=True)
    # Path(VIS_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    # os.makedirs(DA3_OUTPUT_FOLDER, exist_ok=True)
    # os.makedirs(SA2_MASK_FOLDER, exist_ok=True)
    # if ENABLE_SA2_VISUALIZATION:
    #     os.makedirs(SA2_VIS_FOLDER, exist_ok=True)
    # os.makedirs(CLUSTER_MASK_DIR, exist_ok=True)
    # os.makedirs(CLUSTER_CSV_DIR, exist_ok=True)
    # os.makedirs(CLUSTER_VIS_DIR, exist_ok=True)
    # os.makedirs(CLUSTER_POINT_VIS_DIR, exist_ok=True)
    # os.makedirs(SUPERVISION_OUTPUT_DIR, exist_ok=True)
    # for sub in ("depth_png", "depth_npy", "vis_colormap", "masks_npz"):
    #     os.makedirs(os.path.join(SUPERVISION_OUTPUT_DIR, sub), exist_ok=True)
    
    os.makedirs(GS_OUTPUT_DIR, exist_ok=True)    # GSplat 渲染图输出目录


    R_list, t_list = triangulation.load_poses(POSE_PATH)
    num_poses = len(R_list)
    Ps = [triangulation.K @ np.hstack((R_list[i], t_list[i].reshape(3,1))) for i in range(num_poses)]

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[1/4] 初始化 Tracking 系统")
    tracker = tracking.TrackingSystem(device, SUPERPOINT_WEIGHTS, MAX_KEYPOINTS)
    print("[2/4] 加载 DA3 & SAM2")
    da3_model, mask_generator = model.initialize_models(
        device, SA2_CONFIG, SA2_CHECKPOINT, DA3_MODEL_DIR,
        points_per_side=SA2_POINTS_PER_SIDE,
        pred_iou_thresh=SA2_PRED_IOU_THRESH,
        stability_score_thresh=SA2_STABILITY_SCORE_THRESH,
        box_nms_thresh=SA2_BOX_NMS_THRESH,
        crop_n_layers=SA2_CROP_N_LAYERS,
        crop_n_points_downscale_factor=SA2_CROP_N_POINTS_DOWNSCALE_FACTOR,
        min_mask_region_area=SA2_MIN_MASK_REGION_AREA
    )
    print("[3/4] Supervision 模块已就绪")
    print("[4/4] GSplat 模块已就绪")

    if ENABLE_SA2_VISUALIZATION:
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)
    else:
        colors = None

    print("=" * 60)
    print("所有模型加载完成，开始逐帧处理")
    print("=" * 60)

    point_cache = {}

    # 第一帧初始化 Tracking
    first_img_path = os.path.join(IMAGE_FOLDER, f"{START_FRAME:06d}.png")
    if not os.path.exists(first_img_path):
        raise FileNotFoundError(f"第一帧图像不存在: {first_img_path}")
    tracker.initialize_first_frame(first_img_path, START_FRAME)
    print(f"[Frame {START_FRAME:06d}] Tracking 初始化完成，提取 {len(tracker.id_map_prev)} 个特征点")

    # 保存上一帧训练后的高斯状态，供下一帧筛选
    prev_gs_state = None
    sh_dim = (GS_CONFIG.sh_degree + 1) ** 2

    # 逐帧处理
    for frame_id in range(START_FRAME, END_FRAME + 1):
        img_path = os.path.join(IMAGE_FOLDER, f"{frame_id:06d}.png")
        if not os.path.exists(img_path):
            print(f"警告：帧 {frame_id:06d} 图像不存在，跳过")
            continue

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"错误：无法读取图像 {img_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        target_shape = image_rgb.shape

        print(f"\n[{frame_id}/{END_FRAME}] 处理: {os.path.basename(img_path)}")
        print("-" * 60)

        # ------------------- 1. Tracking -------------------
        if frame_id == START_FRAME:
            id_map_curr = tracker.id_map_prev.copy()
            matches_data = None
            print(f"  [Frame {frame_id:06d}] 第0帧Tracking完成")
            # 更新 Tracking 状态（为第1帧做准备）
            if frame_id != START_FRAME:
                tracker.update_state(id_map_curr, matches_data['feats_curr'], matches_data['image_tensor'])
            print(f"{frame_id:06d}处理完成")

            del image_bgr, image_rgb            # 清理

            if matches_data is not None:
                del matches_data
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            continue  # 跳过后续所有处理，进入下一帧
        else:
            image, id_map_curr, matches_data = tracker.process_frame(img_path, frame_id)
            if image is None:
                continue

        # ------------------- 2. Triangulation -------------------
        if frame_id != START_FRAME:
            for pid in set(id_map_curr.values()):
                obs_dict = tracker.get_feature_observations(pid)
                if len(obs_dict) < 2:
                    continue
                frame_ids = sorted(obs_dict.keys())
                valid = [fid for fid in frame_ids if fid < num_poses]
                if len(valid) < 2:
                    continue
                obs_list = [obs_dict[fid] for fid in valid]
                P_list = [Ps[fid] for fid in valid]
                R_obs = [R_list[fid] for fid in valid]
                t_obs = [t_list[fid] for fid in valid]

                X_init = triangulation.linear_triangulate(obs_list, P_list)
                X_opt = triangulation.optimise_point(X_init, obs_list, P_list, triangulation.K)
                omega = triangulation.compute_confidence(obs_list, R_obs, t_obs, X_opt, triangulation.K)
                point_cache[pid] = (X_opt, omega)

        # ------------------- 3. 输出当前帧特征点 CSV [OUTPUT DISABLED] -------------------
        csv_path = frame_csv_dir / f"triangulated_points_{frame_id:06d}.csv"
        # with open(csv_path, 'w', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(['frame_id', 'point_id', 'pixel_x', 'pixel_y', 'X', 'Y', 'Z', 'omega'])
        #     for idx1, pid in id_map_curr.items():
        #         if pid not in point_cache:
        #             continue
        #         X_opt, omega = point_cache[pid]
        #         u, v = tracker.feature_obs[pid][frame_id]
        #         Xc = R_list[frame_id] @ X_opt + t_list[frame_id]
        #         writer.writerow([f"{frame_id:06d}", pid, f"{u:.6f}", f"{v:.6f}",
        #                          float(Xc[0]), float(Xc[1]), float(Xc[2]), f"{omega:.3f}"])

        # ------------------- 4. DA3 + SA2 -------------------
        da3_time, sa2_time, num_masks, depth_path, mask_path = model.process_frame_da3_sa2(
            frame_id, img_path, da3_model, mask_generator,
            device, ENABLE_SA2_VISUALIZATION, colors,
            DA3_OUTPUT_FOLDER, SA2_MASK_FOLDER, SA2_VIS_FOLDER
        )
        # ------------------- 5. Cluster -------------------
        depth_png = os.path.join(DA3_OUTPUT_FOLDER, f"{frame_id:06d}.png")
        mask_npz = os.path.join(SA2_MASK_FOLDER, f"{frame_id:06d}_masks.npz")
        points_csv = str(csv_path)

        cluster_mask_path = None
        cluster_csv_path = None
        if os.path.exists(points_csv) and os.path.exists(depth_png) and os.path.exists(mask_npz):
            try:
                cluster_mask_path, cluster_csv_path = cluster.process_clustering_for_frame(
                    frame_id, depth_png, mask_npz, points_csv,
                    output_mask_dir=CLUSTER_MASK_DIR,
                    output_csv_dir=CLUSTER_CSV_DIR,
                    output_viz_dir=CLUSTER_VIS_DIR,
                    output_point_viz_dir=CLUSTER_POINT_VIS_DIR,
                    image_folder=IMAGE_FOLDER
                )
            except Exception as e:
                print(f"  [Cluster] 帧 {frame_id:06d} 处理失败: {e}")
        else:
            print(f"  [Cluster] 跳过帧 {frame_id:06d}: 缺少必要文件")

        # ------------------- 6. Supervision -------------------
        if cluster_csv_path is not None and os.path.exists(cluster_csv_path):
            try:
                df_current_frame = pd.read_csv(cluster_csv_path)
                required_cols = ['frame_id','mask_id','point_id','pixel_x','pixel_y','X','Y','Z','omega']
                if all(col in df_current_frame.columns for col in required_cols):
                    depth_rel = sv.read_da_rel_depth_png(depth_png)
                    if cluster_mask_path is not None and os.path.exists(cluster_mask_path):
                        bitmaps = sv.load_masks_npz(cluster_mask_path)
                        # [OUTPUT DISABLED] 掩码复制到 Supervision 目录
                        # shutil.copy(cluster_mask_path, os.path.join(SUPERVISION_OUTPUT_DIR, "masks_npz", f"{frame_id:06d}.npz"))
                        _ = sv.process_one_frame(
                            frame=frame_id,
                            df_all=df_current_frame,
                            depth_rel_r=depth_rel,
                            bitmaps=bitmaps,
                            out_dir=SUPERVISION_OUTPUT_DIR,
                            fx=sv.fx, fy=sv.fy, cx=sv.cx, cy=sv.cy,
                            d_max=SV_D_MAX,
                            tau_sky=SV_TAU_SKY,
                            plane_iters=SV_PLANE_ITERS,
                            plane_thresh=SV_PLANE_THRESH,
                            plane_improve_ratio=SV_PLANE_IMPROVE_RATIO,
                            use_plane=SV_PLANE_USE
                        )
                    else:
                        print(f"  [Supervision] 警告：找不到 mask 文件，跳过")
                else:
                    print(f"  [Supervision] 警告：CSV 缺少必要列，跳过")
            except Exception as e:
                print(f"  [Supervision] 帧 {frame_id:06d} 处理失败: {e}")
        else:
            print(f"  [Supervision] 警告：当前帧无 cluster CSV，跳过")

        # ------------------- 7. GSplat（核心逻辑）-------------------
        # 读取绝对深度
        depth = None
        sv_depth_npy = os.path.join(SUPERVISION_OUTPUT_DIR, "depth_npy", f"{frame_id:06d}_abs_depth.npy")
        if os.path.exists(sv_depth_npy):
            depth = np.load(sv_depth_npy).astype(np.float32)
            print(f"  [GSplat] 使用绝对深度")
        else:
            depth_npy_path = os.path.join(DA3_OUTPUT_FOLDER, f"{frame_id:06d}.npy")
            if os.path.exists(depth_npy_path):
                depth = np.load(depth_npy_path).astype(np.float32)
                print(f"  [GSplat] 使用原始深度")
            else:
                print(f"  [GSplat] 警告：无深度数据，跳过该帧")
                continue
        
        if depth is not None:
            depth[(depth < 0.1) | (depth > 1000.0)] = 0

        current_pose = np.eye(4, dtype=np.float64)
        current_pose[:3, :3] = R_list[frame_id]
        current_pose[:3, 3] = t_list[frame_id]

        # ====== 尚未初始化过 GS ======
        if prev_gs_state is None:
            gs_trainer = StreamingGaussianSplatting(
                device=device,
                config=GS_CONFIG,
                strategy_config=GS_STRATEGY,
                camera_config=GS_CAMERA,
            )
            if depth is not None and np.any(depth > 0):
                gs_trainer.initialize_from_depth(
                    image=image_rgb,
                    depth=depth,
                    pose=current_pose
                )
                print(f"  [GS] 第{frame_id}帧初始化完成, 高斯数: {len(gs_trainer.means)}")
            else:
                print(f"  [GS] 警告：第{frame_id}帧无有效深度，跳过 GS")
                continue  # prev_gs_state 保持 None，下一帧继续尝试初始化

        # ====== 已有上一帧GS状态 ======
        else:
            # 1) 评估上一帧高斯在当前帧的颜色误差
            color_errors = evaluate_gaussian_color_errors(
                prev_gs_state, image_rgb, current_pose, GS_CAMERA, device
            )
            
            # 2) 筛选旧高斯
            keep_mask = filter_gaussians(
                prev_gs_state, color_errors, config=FILTER_CONFIG
            )
            
            old_means = prev_gs_state['means'][keep_mask].to(device)
            old_quats = prev_gs_state['quats'][keep_mask].to(device)
            old_scales = prev_gs_state['scales'][keep_mask].to(device)
            old_opacities = prev_gs_state['opacities'][keep_mask].to(device)
            old_colors = prev_gs_state['colors'][keep_mask].to(device)
            
            # 3) 用当前帧绝对深度构造新高斯参数
            new_points, new_colors = extract_new_points_from_depth(
                image_rgb, depth, current_pose, GS_CAMERA, device
            )
            new_params = build_gaussian_params_from_points(new_points, new_colors, sh_dim, device)
            
            # 4) 合并新旧高斯
            if new_params is not None:
                new_means, new_quats, new_scales, new_opacities, new_colors_full = new_params
                combined_means = torch.cat([old_means, new_means], dim=0)
                combined_quats = torch.cat([old_quats, new_quats], dim=0)
                combined_scales = torch.cat([old_scales, new_scales], dim=0)
                combined_opacities = torch.cat([old_opacities, new_opacities], dim=0)
                combined_colors = torch.cat([old_colors, new_colors_full], dim=0)
                n_new = len(new_means)
            else:
                combined_means = old_means
                combined_quats = old_quats
                combined_scales = old_scales
                combined_opacities = old_opacities
                combined_colors = old_colors
                n_new = 0
            
            gs_trainer = StreamingGaussianSplatting(
                device=device,
                config=GS_CONFIG,
                strategy_config=GS_STRATEGY,
                camera_config=GS_CAMERA,
            )
            gs_trainer.means = torch.nn.Parameter(combined_means)
            gs_trainer.quats = torch.nn.Parameter(combined_quats)
            gs_trainer.scales = torch.nn.Parameter(combined_scales)
            gs_trainer.opacities = torch.nn.Parameter(combined_opacities)
            gs_trainer.colors = torch.nn.Parameter(combined_colors)
            gs_trainer.initialized = True
            gs_trainer.step = 0
            gs_trainer.frame_count = frame_id
            gs_trainer.sh_dim = sh_dim
            gs_trainer._setup_optimizer()
            
        # ====== 统一训练 ======
        print(f"  [GS] 训练帧 {frame_id}（{GS_ITERS} 轮）")
        best_loss = float('inf')
        best_round_idx = -1
        best_rendered = None
        
        for round_idx in range(GS_ITERS):
            rendered_raw, loss = gs_trainer.on_new_frame(
                image=image_rgb,
                pose=current_pose,
                depth=None
            )
            
            if loss < best_loss:
                best_loss = loss
                best_round_idx = round_idx
                best_rendered = rendered_raw.copy() if isinstance(rendered_raw, np.ndarray) else np.array(rendered_raw)
            
            if GS_EARLY_STOP_LOSS is not None and loss < GS_EARLY_STOP_LOSS:
                print(f"    [GS] 早停：loss {loss:.6f} < 阈值 {GS_EARLY_STOP_LOSS} @ round {round_idx+1}")
                break
        
        # ====== 渲染并保存当前帧 ======
        if best_rendered is not None:
            rendered_rgb = ensure_rgb_uint8(best_rendered, target_shape=target_shape)
        else:
            rendered_rgb = ensure_rgb_uint8(rendered_raw, target_shape=target_shape)
        
        save_path = os.path.join(GS_OUTPUT_DIR, f"{frame_id:06d}_render.png")
        cv2.imwrite(save_path, cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR))
        print(f"  [GS] 渲染输出: {save_path} (best_loss={best_loss:.6f} @ round {best_round_idx+1}/{GS_ITERS})")
        
        if isinstance(rendered_raw, torch.Tensor):
            raw_np = rendered_raw.detach().cpu().numpy()
        else:
            raw_np = np.array(rendered_raw)

        # ====== 保存状态供下一帧筛选 ======
        prev_gs_state = gs_trainer.get_state()

        # ------------------- 8. 可视化与状态更新 [OUTPUT DISABLED] -------------------
        # if matches_data is not None:
        #     tracker.visualize_matches(
        #         matches_data['image_tensor'],
        #         matches_data,
        #         save_path=os.path.join(VIS_SAVE_DIR, f"{frame_id:06d}.png")
        #     )
        if frame_id != START_FRAME:
            tracker.update_state(id_map_curr, matches_data['feats_curr'], matches_data['image_tensor'])

        print(f"{frame_id:06d}处理完成")

        # 清理显存
        del image_bgr, image_rgb, depth, gs_trainer
        if matches_data is not None:
            del matches_data
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # 最终统计
    print("\n" + "="*60)
    print("处理完成！")
    print(f"GSplat渲染结果: {GS_OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()