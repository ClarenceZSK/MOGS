import os
import time
import gc
import numpy as np
import torch
import cv2
from PIL import Image

from depth_anything_3.api import DepthAnything3
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def visualize_masks(image, anns, colors, alpha=0.5):
    """SA2分割结果可视化"""
    if len(anns) == 0:
        return image.copy()
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    overlay = image.copy().astype(np.float32)
    for idx, ann in enumerate(sorted_anns):
        mask = ann['segmentation']
        color = colors[idx % len(colors)]
        mask_color = np.zeros_like(image, dtype=np.float32)
        mask_color[mask] = color
        overlay[mask] = overlay[mask] * (1 - alpha) + mask_color[mask] * alpha
    
    vis_image = overlay.astype(np.uint8)
    for idx, ann in enumerate(sorted_anns):
        mask = ann['segmentation'].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = colors[idx % len(colors)].tolist()
        cv2.drawContours(vis_image, contours, -1, color, 1)
    return vis_image

def initialize_models(device, sa2_config, sa2_checkpoint, da3_model_dir, 
                      points_per_side=42, pred_iou_thresh=0.82, stability_score_thresh=0.82,
                      box_nms_thresh=0.48, crop_n_layers=0, crop_n_points_downscale_factor=0,
                      min_mask_region_area=0):

    print("[Model] 加载 DA3...")
    da3_model = DepthAnything3.from_pretrained(da3_model_dir).to(device).eval()
    
    print("[Model] 加载 SAM2...")
    torch.backends.cudnn.benchmark = True
    sam2_model = build_sam2(sa2_config, sa2_checkpoint, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        box_nms_thresh=box_nms_thresh,               # 新增可调
        crop_n_layers=crop_n_layers,                 # 新增可调
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,  # 新增可调
        min_mask_region_area=min_mask_region_area,   # 新增可调
    )
    
    return da3_model, mask_generator

def process_frame_da3_sa2(frame_id, img_path, da3_model, mask_generator, 
                          device, enable_vis, colors, 
                          da3_output_folder, sa2_mask_folder, sa2_vis_folder):
    """
    处理单帧的DA3深度估计和SA2分割
    返回: (da3_time, sa2_time, num_masks, depth_path, mask_path)
    """
    base_name = f"{frame_id:06d}"
    
    # 初始化所有返回值
    da3_time = 0.0
    sa2_time = 0.0
    num_masks = 0
    depth_save_path = "未处理"
    mask_save_path = "未处理"
    anns = []
    
    # 读取图像
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        print(f"  警告: 无法读取图像 {img_path}")
        return da3_time, sa2_time, num_masks, "读取失败", "读取失败"
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]
    
    # ==================== DA3 深度估计 ====================
    try:
        torch.cuda.synchronize()
        da3_start = time.time()
        
        prediction = da3_model.inference([img_path], export_format="")
        depth_map = prediction.depth[0]
        
        if depth_map.shape[0] != orig_h or depth_map.shape[1] != orig_w:
            depth_map = cv2.resize(depth_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        depth_min, depth_max = depth_map.min(), depth_map.max()
        depth_normalized = (depth_map - depth_min) / (depth_max - depth_min) * 65535.0
        depth_uint16 = depth_normalized.astype(np.uint16)
        
        os.makedirs(da3_output_folder, exist_ok=True)
        depth_save_path = os.path.join(da3_output_folder, f"{base_name}.png")
        Image.fromarray(depth_uint16).save(depth_save_path)
        
        # 同时保存原始深度npy，供后续模块降级读取
        npy_save_path = os.path.join(da3_output_folder, f"{base_name}.npy")
        np.save(npy_save_path, depth_map.astype(np.float32))
        
        torch.cuda.synchronize()
        da3_time = time.time() - da3_start
        
        del prediction, depth_map, depth_normalized, depth_uint16
        
    except Exception as e:
        print(f"  DA3处理失败: {e}")
        depth_save_path = f"错误: {str(e)}"
    
    # ==================== SA2 分割 ====================
    try:
        torch.cuda.synchronize()
        sa2_start = time.time()
        
        if hasattr(mask_generator.predictor, 'reset_state'):
            mask_generator.predictor.reset_state()
        
        anns = mask_generator.generate(image_rgb)
        num_masks = len(anns)
        
        torch.cuda.synchronize()
        sa2_time = time.time() - sa2_start
        
        os.makedirs(sa2_mask_folder, exist_ok=True)
        mask_save_path = os.path.join(sa2_mask_folder, f"{base_name}_masks.npz")
        
        if anns:
            masks_list = [ann['segmentation'].astype(np.uint8) for ann in anns]
            bitmaps = np.stack(masks_list, axis=0)
            np.savez_compressed(mask_save_path, bitmaps=bitmaps)
            
            del bitmaps, masks_list
        else:
            np.savez_compressed(mask_save_path, bitmaps=np.array([]))
    except Exception as e:
        print(f"  SA2处理失败: {e}")
        mask_save_path = f"错误: {str(e)}"
        anns = []
        num_masks = 0
    
    # 清理显存
    del image_bgr, image_rgb
    if 'anns' in locals():
        del anns
    gc.collect()
    torch.cuda.empty_cache()
    
    return da3_time, sa2_time, num_masks, depth_save_path, mask_save_path