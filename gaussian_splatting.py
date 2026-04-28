#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流式 3D Gaussian Splatting 训练器
基于 gsplat 1.5.3，支持逐帧接收前端图像并实时训练
支持 sh_degree > 0（球谐函数）
"""

import os
import time
import gc
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from PIL import Image

from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

K = np.array([
    [7.188560000000e+02, 0, 6.071928000000e+02],
    [0, 7.188560000000e+02, 1.852157000000e+02],
    [0, 0, 1]
], dtype=np.float64)


@dataclass
class CameraParams:
    width: int = 1241
    height: int = 376
    fx: float = 718.856
    fy: float = 718.856
    cx: float = 607.1928
    cy: float = 185.2157
    
    def get_intrinsics_matrix(self, device="cuda"):
        K = torch.tensor([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
        return K


@dataclass  
class GaussianParams:
    sh_degree: int = 0
    init_opacity: float = 0.1
    init_scale: float = 1.0
    scale_min: float = 0.01
    scale_max: float = 10.0
    
    lr_means: float = 1.6e-4
    lr_quats: float = 1.6e-3
    lr_scales: float = 5e-3
    lr_opacities: float = 5e-2
    lr_sh: float = 1.6e-3
    
    loss_l1_weight: float = 0.8
    loss_ssim_weight: float = 0.2
    
    lr_decay_rate: float = 0.97
    lr_decay_interval: int = 100
    
    scale_clamp_min: float = 0.001
    scale_clamp_max: float = 0.1
    
    near_plane: float = 0.01
    far_plane: float = 100.0


@dataclass
class StrategyParams:
    verbose: bool = False
    prune_opa: float = 0.005          # 剪枝不透明度阈值
    grow_grad2d: float = 0.0002       # 分裂梯度阈值
    grow_scale3d: float = 0.01        # 分裂尺度阈值
    max_gaussians: int = 200000       # 【新增】高斯数量上限，防爆显存
    split_scale_factor: float = 1.6   # 【新增】分裂后尺度缩放因子
    densify_interval: int = 1000      # 【新增】每隔多少步执行一次密度控制


class StreamingGaussianSplatting:
    def __init__(self, device="cuda", config: Optional[GaussianParams] = None,
                 strategy_config: Optional[StrategyParams] = None,
                 camera_config: Optional[CameraParams] = None):
        self.device = torch.device(device)
        self.config = config or GaussianParams()
        self.strategy_config = strategy_config or StrategyParams()
        self.camera = camera_config or CameraParams()
        
        # 球谐维度
        self.sh_dim = (self.config.sh_degree + 1) ** 2
        
        self.step = 0
        self.initialized = False
        
        self.means = nn.Parameter(torch.empty(0, 3, device=self.device))
        self.quats = nn.Parameter(torch.empty(0, 4, device=self.device))
        self.scales = nn.Parameter(torch.empty(0, 3, device=self.device))
        self.opacities = nn.Parameter(torch.empty(0, device=self.device))
        self.colors = nn.Parameter(torch.empty(0, self.sh_dim, 3, device=self.device))
        
        self.optimizer = None
        self._setup_optimizer()
        
        self.strategy = DefaultStrategy(
            verbose=self.strategy_config.verbose,
            prune_opa=self.strategy_config.prune_opa,
            grow_grad2d=self.strategy_config.grow_grad2d,
            grow_scale3d=self.strategy_config.grow_scale3d,
        )
        
        self.background = torch.zeros(3, device=self.device)
        self.frame_count = 0
        self.total_loss = 0.0
            
    def _setup_optimizer(self):
        self.optimizer = Adam([
            {'params': [self.means], 'lr': self.config.lr_means, 'name': 'means'},
            {'params': [self.quats], 'lr': self.config.lr_quats, 'name': 'quats'},
            {'params': [self.scales], 'lr': self.config.lr_scales, 'name': 'scales'},
            {'params': [self.opacities], 'lr': self.config.lr_opacities, 'name': 'opacities'},
            {'params': [self.colors], 'lr': self.config.lr_sh, 'name': 'colors'},
        ])
    
    def _build_viewmat(self, pose: np.ndarray) -> torch.Tensor:
        if pose.shape == (3, 4):
            pose = np.vstack([pose, [0, 0, 0, 1]])
        viewmat = torch.from_numpy(pose).float().to(self.device)
        return viewmat
    
    def initialize_from_depth(self, image: np.ndarray, depth: np.ndarray, 
                              pose: np.ndarray, mask: Optional[np.ndarray] = None):        
        H, W = image.shape[:2]
        image_tensor = torch.from_numpy(image).float().to(self.device) / 255.0
        depth_tensor = torch.from_numpy(depth).float().to(self.device)
        
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u = torch.from_numpy(u).float().to(self.device)
        v = torch.from_numpy(v).float().to(self.device)
        
        Z = depth_tensor
        X = (u - self.camera.cx) * Z / self.camera.fx
        Y = (v - self.camera.cy) * Z / self.camera.fy
        
        points_3d = torch.stack([X, Y, Z], dim=-1)
        
        valid_mask = (Z > 0.1) & (Z < 100.0) & torch.isfinite(Z)
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).bool().to(self.device)
            valid_mask &= mask_tensor
        
        valid_points = points_3d[valid_mask]
        valid_colors = image_tensor[valid_mask]  # [N, 3]
        
        target_num = min(100000, len(valid_points))
        if len(valid_points) > target_num:
            indices = torch.randperm(len(valid_points))[:target_num]
            valid_points = valid_points[indices]
            valid_colors = valid_colors[indices]
        
        num_points = len(valid_points)
        
        # 构建球谐颜色: [N, sh_dim, 3]，只有 DC 分量有值
        colors_full = torch.zeros(num_points, self.sh_dim, 3, device=self.device)
        colors_full[:, 0, :] = valid_colors
        
        self.means = nn.Parameter(valid_points)
        self.quats = nn.Parameter(self._random_quats(num_points))
        self.scales = nn.Parameter(torch.ones(num_points, 3, device=self.device) * 0.01)
        self.opacities = nn.Parameter(torch.ones(num_points, device=self.device) * self.config.init_opacity)
        self.colors = nn.Parameter(colors_full)
        
        self._setup_optimizer()
        self.initialized = True
        self.frame_count = 1
            
    def initialize_from_points(self, points: np.ndarray, colors: np.ndarray):
        print(f"[GSplat] 从点云初始化，点数: {len(points)}")
        
        num_points = len(points)
        points_t = torch.from_numpy(points).float().to(self.device)
        colors_t = torch.from_numpy(colors).float().to(self.device)  # [N, 3]
        
        # 扩展到球谐
        colors_full = torch.zeros(num_points, self.sh_dim, 3, device=self.device)
        colors_full[:, 0, :] = colors_t
        
        self.means = nn.Parameter(points_t)
        self.quats = nn.Parameter(self._random_quats(num_points))
        self.scales = nn.Parameter(torch.ones(num_points, 3, device=self.device) * 0.01)
        self.opacities = nn.Parameter(torch.ones(num_points, device=self.device) * 0.5)
        self.colors = nn.Parameter(colors_full)
        
        self._setup_optimizer()
        self.initialized = True
        self.frame_count = 1    
    def on_new_frame(self, image: np.ndarray, pose: np.ndarray, 
                     depth: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        if not self.initialized:
            if depth is not None:
                self.initialize_from_depth(image, depth, pose)
            else:
                raise ValueError("第一帧必须提供深度图用于初始化")
        
        image_gt = torch.from_numpy(image).float().to(self.device) / 255.0
        viewmat = self._build_viewmat(pose)
        K = self.camera.get_intrinsics_matrix(self.device)
        
        colors_reshaped = torch.sigmoid(self.colors)  # [N, sh_dim, 3]
        opacities = torch.sigmoid(self.opacities)
    
        H, W = self.camera.height, self.camera.width
        backgrounds = self.background[None, :]

        rendered, alpha, meta = rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.scales.clamp(self.config.scale_clamp_min, self.config.scale_clamp_max),
            opacities=opacities,
            colors=colors_reshaped,
            viewmats=viewmat[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.config.sh_degree,
            backgrounds=backgrounds,
            near_plane=self.config.near_plane,
            far_plane=self.config.far_plane,
            render_mode='RGB',
            packed=False,
        )
        rendered = rendered[0]
        
        loss = self._compute_loss(rendered, image_gt)
        # 在 loss.backward() 之前，给高频球谐系数加L2正则
        if self.config.sh_degree > 0:
            # DC分量（第0个）保留，高频（第1~15个）惩罚
            high_freq = self.colors[:, 1:, :]  # [N, 15, 3]
            sh_reg = (high_freq ** 2).mean()
            loss = loss + 0.001 * sh_reg  # 高频惩罚
        loss.backward()
        
        if self.step % self.strategy_config.densify_interval == 0 and self.step > 0:
            self._densify_and_prune(meta)

        #if self.step % 100 == 0 and self.step > 0:
        #    self._densify_and_prune(meta)
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._update_learning_rate()
        
        self.step += 1
        self.frame_count += 1
        self.total_loss += loss.item()

        rendered_np = (rendered.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return rendered_np, loss.item()
        
    def _compute_loss(self, rendered, gt):
        #"""计算损失：L1 + SSIM + 尺度正则"""
        l1_loss = torch.abs(rendered - gt).mean()
        
        # SSIM 损失
        ssim_loss = torch.tensor(0.0, device=rendered.device)
        if self.config.loss_ssim_weight > 0:
            try:
                from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
                ssim = SSIM(data_range=1.0).to(rendered.device)
                rendered_perm = rendered.permute(2, 0, 1).unsqueeze(0)
                gt_perm = gt.permute(2, 0, 1).unsqueeze(0)
                ssim_val = ssim(rendered_perm, gt_perm)
                ssim_loss = 1.0 - ssim_val
            except Exception as e:
                print(f"  [GS] SSIM 失败: {e}")
        
        # 尺度正则化（防止高斯拉成细长条）
        scale_reg = torch.tensor(0.0, device=rendered.device)
        if len(self.scales) > 0:
            scales = self.scales.clamp(min=1e-4)
            max_s = scales.max(dim=-1)[0]
            min_s = scales.min(dim=-1)[0]
            ratio = max_s / (min_s + 1e-6)
            # 只惩罚比值 > 10 的，且用 soft 惩罚
            scale_reg = ((ratio - 10.0).clamp(min=0) ** 2).mean()
        
        # 组合
        loss = (self.config.loss_l1_weight * l1_loss +
                self.config.loss_ssim_weight * ssim_loss +
                0.001 * scale_reg)  # 尺度正则权重 0.001
        
        return loss
    
 
    #def _densify_and_prune(self, meta: dict):
    #    render_grads = meta.get("render_grads", None)
    #   if render_grads is None:
    #        return
    #    
    #    with torch.no_grad():
    #       opacity_mask = torch.sigmoid(self.opacities) > self.strategy_config.prune_opa
    #        if opacity_mask.sum() < len(self.opacities):
    #            self._prune_points(~opacity_mask)
    #        
    #        if self.step % 300 == 0:
    #            pass
   

    def _densify_and_prune(self, meta: dict):
        """
        密度控制：剪枝低不透明度高斯 + 分裂大尺度/高梯度高斯
        """
        with torch.no_grad():
            # ===== 1. 剪枝：去掉几乎透明的高斯 =====
            opacity_mask = torch.sigmoid(self.opacities) > self.strategy_config.prune_opa
            if opacity_mask.sum() < len(self.opacities):
                n_prune = (~opacity_mask).sum().item()
                self._prune_points(~opacity_mask)
            # ===== 2. 分裂：梯度大 或 尺度大的高斯 =====
            current_n = len(self.means)
            max_n = self.strategy_config.max_gaussians
            if current_n >= max_n:
                return  # 已达上限，不再分裂

            # means.grad 在 backward() 后已有值，直接取 3D 梯度范数近似
            if self.means.grad is None:
                return
            
            grad_norm = self.means.grad.norm(dim=-1)           # [N]
            scale_norm = self.scales.norm(dim=-1)              # [N]
            
            # 分裂条件：view-space 梯度大 或 3D 尺度太大
            split_mask = (grad_norm > self.strategy_config.grow_grad2d) | \
                         (scale_norm > self.strategy_config.grow_scale3d)
            
            n_split = split_mask.sum().item()
            if n_split == 0:
                return
            
            # 限制分裂数量，防止超过上限（1个分裂成2个，占2个名额）
            max_new = max_n - current_n
            max_splits = max_new // 2
            if n_split > max_splits and max_splits > 0:
                scores = grad_norm + scale_norm
                _, indices = torch.topk(scores, k=max_splits)
                split_mask = torch.zeros_like(split_mask)
                split_mask[indices] = True
                n_split = max_splits
            
            if n_split > 0:
                self._split_points(split_mask)

    def _split_points(self, mask: torch.Tensor):
        """
        高斯分裂：1 个父高斯 → 2 个子高斯
        - 位置沿 scale 加权方向扰动
        - 尺度缩小
        - 颜色/旋转/不透明度继承
        """
        n = mask.sum().item()
        if n == 0:
            return
        
        # ---- 未分裂的高斯 ----
        means_rest   = self.means[~mask].detach()
        quats_rest   = self.quats[~mask].detach()
        scales_rest  = self.scales[~mask].detach()
        opacities_rest = self.opacities[~mask].detach()
        colors_rest  = self.colors[~mask].detach()
        
        # ---- 被分裂的父高斯 ----
        means_s   = self.means[mask].detach()          # [n, 3]
        quats_s   = self.quats[mask].detach()          # [n, 4]
        scales_s  = self.scales[mask].detach()         # [n, 3]
        opas_s    = self.opacities[mask].detach()      # [n]
        colors_s  = self.colors[mask].detach()         # [n, sh_dim, 3]
        
        # ---- 子高斯 1 & 2 ----
        # 扰动方向：按 scale 各向异性加权，大尺度轴扰动大
        noise = torch.randn_like(means_s) * scales_s * 0.5
        means_c1 = means_s + noise
        means_c2 = means_s - noise
        
        # 尺度缩小（经验值 1.6，两个小球体积≈原球）
        scales_c = scales_s / self.strategy_config.split_scale_factor
        
        # 不透明度继承（也可适当降低，如 opas_s - 0.05）
        opas_c = opas_s
        
        # ---- 拼接回全局参数 ----
        self.means     = nn.Parameter(torch.cat([means_rest, means_c1, means_c2], dim=0))
        self.quats     = nn.Parameter(torch.cat([quats_rest, quats_s, quats_s], dim=0))
        self.scales    = nn.Parameter(torch.cat([scales_rest, scales_c, scales_c], dim=0))
        self.opacities = nn.Parameter(torch.cat([opacities_rest, opas_c, opas_c], dim=0))
        self.colors    = nn.Parameter(torch.cat([colors_rest, colors_s, colors_s], dim=0))
        
        self._setup_optimizer()    
    def _prune_points(self, mask: torch.Tensor):
        self.means = nn.Parameter(self.means[~mask].detach())
        self.quats = nn.Parameter(self.quats[~mask].detach())
        self.scales = nn.Parameter(self.scales[~mask].detach())
        self.opacities = nn.Parameter(self.opacities[~mask].detach())
        self.colors = nn.Parameter(self.colors[~mask].detach())
        self._setup_optimizer()
    
    def _update_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'means':
                param_group['lr'] = (self.config.lr_means * 
                    (self.config.lr_decay_rate ** (self.step / self.config.lr_decay_interval)))
    
    @staticmethod
    def _random_quats(n: int, device="cuda") -> torch.Tensor:
        u1, u2, u3 = torch.rand(3, n, device=device)
        quats = torch.stack([
            torch.sqrt(1 - u1) * torch.sin(2 * np.pi * u2),
            torch.sqrt(1 - u1) * torch.cos(2 * np.pi * u2),
            torch.sqrt(u1) * torch.sin(2 * np.pi * u3),
            torch.sqrt(u1) * torch.cos(2 * np.pi * u3),
        ], dim=-1)
        return quats
    
    def export_ply(self, path: str):
        from plyfile import PlyData, PlyElement
        
        means = self.means.detach().cpu().numpy()
        colors = torch.sigmoid(self.colors).detach().cpu().numpy()  # [N, sh_dim, 3]
        opacities = torch.sigmoid(self.opacities).detach().cpu().numpy()
        
        vertex = np.zeros(len(means), dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('opacity', 'f4'),
        ])
        vertex['x'] = means[:, 0]
        vertex['y'] = means[:, 1]
        vertex['z'] = means[:, 2]
        # 取 DC 分量（第一个球谐系数）
        vertex['red'] = (colors[:, 0, 0] * 255).clip(0, 255).astype(np.uint8)
        vertex['green'] = (colors[:, 0, 1] * 255).clip(0, 255).astype(np.uint8)
        vertex['blue'] = (colors[:, 0, 2] * 255).clip(0, 255).astype(np.uint8)
        vertex['opacity'] = opacities
        
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el]).write(path)
        print(f"[GSplat] 导出 PLY: {path}, 点数: {len(means)}")
    
    def get_state(self) -> dict:
        return {
            'means': self.means.detach().cpu(),
            'quats': self.quats.detach().cpu(),
            'scales': self.scales.detach().cpu(),
            'opacities': self.opacities.detach().cpu(),
            'colors': self.colors.detach().cpu(),
            'step': self.step,
            'frame_count': self.frame_count,
            'sh_dim': self.sh_dim,
        }
    
    def load_state(self, state: dict):
        self.means = nn.Parameter(state['means'].to(self.device))
        self.quats = nn.Parameter(state['quats'].to(self.device))
        self.scales = nn.Parameter(state['scales'].to(self.device))
        self.opacities = nn.Parameter(state['opacities'].to(self.device))
        self.colors = nn.Parameter(state['colors'].to(self.device))
        self.step = state['step']
        self.frame_count = state['frame_count']
        self.sh_dim = state.get('sh_dim', self.sh_dim)
        self._setup_optimizer()
        self.initialized = True