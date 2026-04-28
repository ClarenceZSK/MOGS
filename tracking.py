import numpy as np
import torch
import cv2
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd, filter_features_with_prosac
from lightglue import viz2d

def point_nums(track_counts):
    """统计不同跟踪长度的点数量（用于可视化颜色分级）"""
    track_counts = track_counts.flatten()
    blue = red = yellow = pink = black = 0
    for count in track_counts:
        if 10 < count <= 20:
            blue += 1
        elif 20 < count <= 30:
            red += 1
        elif 30 < count <= 40:
            yellow += 1
        elif 40 < count <= 50:
            pink += 1
        elif count > 50:
            black += 1
    return blue, red, yellow, pink, black

class TrackingSystem:
    """
    封装 SuperPoint + LightGlue 的完整追踪系统
    """
    def __init__(self, device, superpoint_weights, max_keypoints=1024):
        self.device = device
        self.max_keypoints = max_keypoints
        
        print("[Tracking] 加载 SuperPoint...")
        self.extractor = SuperPoint(
            max_num_keypoints=max_keypoints, 
            weights=superpoint_weights
        ).eval().to(device)
        
        print("[Tracking] 加载 LightGlue...")
        self.matcher = LightGlue(features="superpoint").eval().to(device)
        
        # 状态变量
        self.feats_prev = None
        self.image_prev_tensor = None
        self.id_map_prev = {}
        self.next_pid = 0
        self.feature_obs = {}  # pid -> {frame_id: (u, v)}
        self.frame_id = None
        
    def initialize_first_frame(self, image_path, frame_id=0):
        """初始化第一帧，建立初始特征点"""
        self.frame_id = frame_id
        
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        
        image_tensor = load_image(image)
        feats = self.extractor.extract(image_tensor.to(self.device))
        feats_rbd = rbd(feats)
        kpts = feats_rbd["keypoints"]
        
        # 分配初始ID
        self.id_map_prev = {i: pid for i, pid in enumerate(range(kpts.shape[0]))}
        self.next_pid = kpts.shape[0]
        
        # 记录观测
        for idx, pid in self.id_map_prev.items():
            x, y = kpts[idx].cpu().numpy()
            self.feature_obs[pid] = {frame_id: (float(x), float(y))}
        
        # 保存状态
        self.feats_prev = feats
        self.image_prev_tensor = image_tensor
        self.feats_prev["tracking_count"] = torch.zeros(
            (1, kpts.shape[0]), dtype=torch.int, device=self.device
        )
        
        return image, kpts.cpu().numpy()
    
    def process_frame(self, image_path, frame_id):
        """
        处理新一帧，返回匹配结果和观测数据
        返回: (image, id_map_curr, matches_data)
        """
        self.frame_id = frame_id
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"警告：帧 {frame_id:06d} 图像不存在，跳过")
            return None, None, None
        
        image_tensor = load_image(image)
        
        # 特征提取
        feats_curr = self.extractor.extract(image_tensor.to(self.device))
        
        # 匹配
        matches01 = self.matcher({"image0": self.feats_prev, "image1": feats_curr})
        feats0_m, feats1_m, matches01 = [rbd(x) for x in [self.feats_prev, feats_curr, matches01]]
        kpts0 = feats0_m["keypoints"]
        kpts1 = feats1_m["keypoints"]
        matches = matches01["matches"]
        
        # PROSAC过滤
        matches, F = filter_features_with_prosac(kpts0, kpts1, matches, threshold=1.0)
        
        # 更新跟踪计数
        if "tracking_count" not in feats_curr:
            feats_curr["tracking_count"] = torch.zeros(
                (1, kpts1.shape[0]), dtype=torch.int, device=self.device
            )
        for (a, b) in matches:
            feats_curr["tracking_count"][0, b] = self.feats_prev["tracking_count"][0, a] + 1
        
        # 分配当前帧ID
        id_map_curr = {}
        for a, b in matches:
            id_map_curr[b] = self.id_map_prev[a]
        for idx1 in range(kpts1.shape[0]):
            if idx1 not in id_map_curr:
                id_map_curr[idx1] = self.next_pid
                self.next_pid += 1
        
        # 记录观测
        for idx1, pid in id_map_curr.items():
            x, y = kpts1[idx1].cpu().numpy()
            if pid not in self.feature_obs:
                self.feature_obs[pid] = {}
            self.feature_obs[pid][frame_id] = (float(x), float(y))
        
        # 准备返回数据
        matches_data = {
            'kpts0': kpts0,
            'kpts1': kpts1,
            'matches': matches,
            'tracking_count': feats_curr["tracking_count"],
            'feats_curr': feats_curr,
            'image_tensor': image_tensor
        }
        
        return image, id_map_curr, matches_data
    
    def update_state(self, id_map_curr, feats_curr, image_tensor):
        """更新状态为下一帧做准备"""
        self.id_map_prev = id_map_curr
        self.feats_prev = feats_curr
        self.image_prev_tensor = image_tensor
    
    def get_feature_observations(self, pid):
        """获取某个特征点的所有观测"""
        return self.feature_obs.get(pid, {})
    
    def get_all_observations(self):
        """获取所有特征点的观测"""
        return self.feature_obs
    
    def visualize_matches(self, image_tensor, matches_data, save_path=None):
        """
        生成LightGlue可视化图像
        """
        kpts0 = matches_data['kpts0']
        kpts1 = matches_data['kpts1']
        matches = matches_data['matches']
        tracking_count = matches_data['tracking_count']
        
        m_kpts0 = kpts0[matches[..., 0]]
        m_kpts1 = kpts1[matches[..., 1]]
        m_tracking_count = tracking_count[0, matches[..., 1]]
        
        # 绘制匹配
        plot_image = image_tensor
        viz2d.plot_matches_single_image(
            plot_image.permute(1,2,0).cpu().numpy(),
            m_kpts0, m_kpts1, color=(0,255,0), lw=1
        )
        
        # 绘制关键点
        plot_image = viz2d.draw_keypoints(
            plot_image.permute(1,2,0).cpu().numpy(),
            m_kpts1.cpu().numpy(),
            m_tracking_count.cpu().numpy()
        )
        
        # 转换为BGR
        img_bgr = (plot_image * 255).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        
        # 添加文字信息
        match_rate = len(m_kpts1) / self.max_keypoints * 100
        blue, red, yellow, pink, black = point_nums(m_tracking_count.cpu().numpy())
        
        cv2.putText(img_bgr, f"Frame {self.frame_id}  match rate: {match_rate:.1f}%", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(img_bgr, f"blue:{blue} red:{red} yellow:{yellow} pink:{pink} black:{black}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        return img_bgr