import os
import numpy as np
import pandas as pd
import cv2

# ======================== 模块级参数（可由 main.py 覆盖） ========================
D_MAX                = 100.0    # 最大深度（米）
TAU_SKY              = 0.01     # 天空mask的相对深度阈值
PLANE_USE            = True     # 是否启用平面拟合优化
PLANE_ITERS          = 200      # RANSAC迭代次数
PLANE_THRESH         = 0.10     # 平面内点距离阈值（米）
PLANE_IMPROVE_RATIO  = 0.15     # 平面拟合需优于仿射拟合的比例
EPS = 1e-6                      # 数值稳定性常量

# 相机内参（KITTI数据集，可由 main.py 覆盖）
K = np.array([[718.856,   0.   , 607.1928],
              [  0.   , 718.856, 185.2157],
              [  0.   ,   0.   ,   1.    ]], dtype=np.float64)
fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]


# ======================== 工具函数（完整保留） ========================
def read_da_rel_depth_png(path):
    """读取DepthAnything输出的相对深度PNG，归一化到[0,1]"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read depth png: {path}")
    img = img.astype(np.float32)
    mn = float(np.min(img))
    mx = float(np.max(img))
    if mx - mn < EPS:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn + EPS)

def load_masks_npz(npz_path):
    """加载mask的npz文件，提取二值mask"""
    data = np.load(npz_path)
    if 'arr_0' in data:
        m = data['arr_0']
    elif 'bitmaps' in data:
        m = data['bitmaps']
    else:
        key = list(data.keys())[0]
        m = data[key]
    return (m > 0).astype(np.bool_)

def make_mask_id_map(bitmaps):
    """生成像素→mask ID的映射矩阵"""
    K, H, W = bitmaps.shape
    mid = -np.ones((H, W), dtype=np.int32)
    for k in range(K):
        mid[bitmaps[k]] = k
    return mid

def sample_at_pixels(arr, xs, ys):
    """根据像素坐标采样数组值，过滤越界坐标"""
    H, W = arr.shape[:2]
    xs = np.asarray(xs).astype(np.int32)
    ys = np.asarray(ys).astype(np.int32)
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    vals = np.zeros_like(xs, dtype=np.float32)
    vals[ok] = arr[ys[ok], xs[ok]]
    return vals, ok

def fit_affine_weighted(r, d, w):
    """加权拟合仿射模型 d = s*r + t（最小二乘）"""
    r = np.asarray(r, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    ok = np.isfinite(r) & np.isfinite(d) & np.isfinite(w) & (w > 0)
    r, d, w = r[ok], d[ok], w[ok]
    if r.size < 2:
        return None, None, False
    A = np.stack([r, np.ones_like(r)], axis=1)
    W = np.diag(w)
    ATA = A.T @ W @ A
    if np.linalg.cond(ATA) > 1e12:
        return None, None, False
    ATd = A.T @ W @ d
    sol = np.linalg.solve(ATA, ATd)
    return float(sol[0]), float(sol[1]), True

def ransac_plane(points, n_iters=200, thresh=0.1, rng=None):
    """RANSAC拟合3D平面（ax+by+cz+d=0）"""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        return None, None, None
    if rng is None:
        rng = np.random.default_rng(42)
    best = (None, None, None)  # plane,inl,rmse
    for _ in range(n_iters):
        idx = rng.choice(pts.shape[0], 3, replace=False)
        p1,p2,p3 = pts[idx]
        n = np.cross(p2-p1, p3-p1)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -np.dot(n, p1)
        dist = np.abs(pts @ n + d)
        inl = dist < thresh
        if inl.sum() < 3:
            continue
        q = pts[inl]
        ctr = q.mean(axis=0)
        _,_,Vt = np.linalg.svd(q - ctr, full_matrices=False)
        n2 = Vt[-1]
        n2 = n2 / (np.linalg.norm(n2) + EPS)
        d2 = -np.dot(n2, ctr)
        dist2 = np.abs(pts @ n2 + d2)
        inl2 = dist2 < thresh
        if inl2.sum() == 0:
            continue
        rmse2 = float(np.sqrt(np.mean(dist2[inl2]**2)))
        if best[0] is None or (inl2.sum() > best[1].sum()) or \
           (inl2.sum() == best[1].sum() and rmse2 < best[2]):
            best = ((n2.astype(np.float64), float(d2)), inl2, rmse2)
    return best

def plane_depth_map(plane, fx, fy, cx, cy, H, W, d_max=100.0):
    """根据拟合的3D平面，生成全图像的深度图（Z值）"""
    n, d = plane
    us = np.arange(W, dtype=np.float32)[None, :].repeat(H, 0)
    vs = np.arange(H, dtype=np.float32)[:, None].repeat(W, 1)
    x = (us - cx) / (fx + EPS)
    y = (vs - cy) / (fy + EPS)
    dir_dot = n[0]*x + n[1]*y + n[2]*1.0
    z = -d / (dir_dot + EPS)
    z[(dir_dot > -1e-6) | (z <= 0)] = np.nan
    z[z > d_max] = d_max
    return z.astype(np.float32)

def weighted_rmse(pred_at_pts, gt_z, w):
    """计算加权RMSE（评估拟合精度）"""
    ok = np.isfinite(pred_at_pts) & np.isfinite(gt_z) & (w > 0)
    if ok.sum() == 0:
        return np.inf
    e2 = (pred_at_pts[ok] - gt_z[ok])**2
    ww = w[ok]
    ww = ww / (ww.sum() + EPS)
    return float(np.sqrt(np.sum(ww * e2)))

def per_mask_centers(bitmaps, valid_mids):
    """计算有效mask的中心坐标"""
    centers = {}
    for mid in valid_mids:
        ys, xs = np.where(bitmaps[mid])
        if xs.size:
            centers[mid] = (float(xs.mean()), float(ys.mean()))
    return centers

def nearest_mask_param_fill(H, W, centers, s_dict, t_dict):
    """为缺失深度的像素，按“最近mask中心”填充仿射参数(s/t)"""
    s_map = np.full((H, W), np.nan, dtype=np.float32)
    t_map = np.full((H, W), np.nan, dtype=np.float32)
    if not centers:
        return s_map, t_map
    mids = np.array(list(centers.keys()), dtype=np.int32)
    C = np.array([centers[m] for m in mids], dtype=np.float32)  # [M,2]
    us = np.arange(W, dtype=np.float32)[None, :].repeat(H, 0)
    vs = np.arange(H, dtype=np.float32)[:, None].repeat(W, 1)
    du = us[..., None] - C[None, None, :, 0]
    dv = vs[..., None] - C[None, None, :, 1]
    nn = np.argmin(du*du + dv*dv, axis=2)  # [H,W] -> idx
    s_arr = np.array([s_dict[m] for m in mids], dtype=np.float32)
    t_arr = np.array([t_dict[m] for m in mids], dtype=np.float32)
    s_map = s_arr[nn]
    t_map = t_arr[nn]
    return s_map, t_map

def write_depth_outputs(out_dir, frame, depth_m, d_max=100.0):
    """输出绝对深度图（npy+png格式）"""
    npy_dir = os.path.join(out_dir, "depth_npy")
    png_dir = os.path.join(out_dir, "depth_png")
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    depth_m = np.clip(np.asarray(depth_m, dtype=np.float32), 0.0, d_max)
    np.save(os.path.join(npy_dir, f"{frame:06d}_abs_depth.npy"), depth_m)
    depth_cm = np.round(depth_m * 100.0).astype(np.uint16)
    cv2.imwrite(os.path.join(png_dir, f"{frame:06d}_abs_depth.png"), depth_cm)

def write_colormap_vis(out_dir, frame, depth_m, d_max, bitmaps):
    """生成深度图的彩色可视化（INFERNO色板），并标注每个mask的平均深度"""
    vis_dir = os.path.join(out_dir, "vis_colormap")
    os.makedirs(vis_dir, exist_ok=True)

    dm = np.asarray(depth_m, dtype=np.float32)
    dm = np.clip(dm, 0.0, d_max)

    finite = np.isfinite(dm)
    if np.any(finite):
        p1 = float(np.nanpercentile(dm[finite], 1.0))
        p99 = float(np.nanpercentile(dm[finite], 99.0))
        if p99 - p1 < 1e-6:
            p1, p99 = 0.0, d_max
    else:
        p1, p99 = 0.0, d_max

    norm = (dm - p1) / (p99 - p1 + 1e-6)
    norm = np.clip(norm, 0.0, 1.0)
    vis_u8 = (norm * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(vis_u8, cv2.COLORMAP_INFERNO)

    K = bitmaps.shape[0]
    for mid in range(K):
        m = bitmaps[mid]
        if m.sum() == 0:
            continue
        d_mean = float(np.nanmean(dm[m]))
        ys, xs = np.where(m)
        cx, cy = int(xs.mean()), int(ys.mean())
        text = f"{d_mean:.1f}m"
        cv2.putText(vis, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    cv2.imwrite(os.path.join(vis_dir, f"{frame:06d}_vis.png"), vis)


# ======================== 核心单帧处理函数（完整保留，供main调用） ========================
def process_one_frame(frame, df_all, depth_rel_r, bitmaps, out_dir,
                      fx=None, fy=None, cx=None, cy=None,
                      d_max=None, tau_sky=None,
                      plane_iters=None, plane_thresh=None,
                      plane_improve_ratio=None, use_plane=None):

    # 使用模块默认值（若未传参）
    fx = fx if fx is not None else globals()['fx']
    fy = fy if fy is not None else globals()['fy']
    cx = cx if cx is not None else globals()['cx']
    cy = cy if cy is not None else globals()['cy']
    d_max = d_max if d_max is not None else D_MAX
    tau_sky = tau_sky if tau_sky is not None else TAU_SKY
    plane_iters = plane_iters if plane_iters is not None else PLANE_ITERS
    plane_thresh = plane_thresh if plane_thresh is not None else PLANE_THRESH
    plane_improve_ratio = plane_improve_ratio if plane_improve_ratio is not None else PLANE_IMPROVE_RATIO
    use_plane = use_plane if use_plane is not None else PLANE_USE

    r = depth_rel_r  # HxW, float32
    H, W = r.shape
    K = bitmaps.shape[0]
    mask_id_map = make_mask_id_map(bitmaps)

    df = df_all[df_all['frame_id'] == frame].copy()
    df = df[np.isfinite(df['Z']) & (df['Z'] > 0) & (df['Z'] <= d_max)]

    df['px'] = df['pixel_x'].astype(int)
    df['py'] = df['pixel_y'].astype(int)
    r_i, ok_pix = sample_at_pixels(r, df['px'].values, df['py'].values)
    df = df[ok_pix].copy()
    r_i = r_i[ok_pix]
    df['r'] = r_i

    depth_affine = np.full((H, W), np.nan, dtype=np.float32)
    depth_final  = np.full((H, W), np.nan, dtype=np.float32)

    s_dict, t_dict = {}, {}
    sky_masks = set()
    used_plane_masks = set()
    logs = []

    for mid in range(K):
        area = bitmaps[mid].sum()
        if area == 0:
            logs.append(dict(frame=frame, mask_id=mid, n_pts=0, sky=0,
                             s=np.nan,t=np.nan,eps1=np.nan,plane_used=0,eps2=np.nan,inliers=0,plane_rmse=np.nan))
            continue
        mean_r = float(r[bitmaps[mid]].mean())
        if mean_r < tau_sky:
            sky_masks.add(mid)
            depth_affine[bitmaps[mid]] = d_max
            depth_final[bitmaps[mid]]  = d_max
            logs.append(dict(frame=frame, mask_id=mid, n_pts=0, sky=1,
                             s=np.nan,t=np.nan,eps1=np.nan,plane_used=0,eps2=np.nan,inliers=0,plane_rmse=np.nan))
        else:
            logs.append(dict(frame=frame, mask_id=mid, n_pts=0, sky=0,
                             s=np.nan,t=np.nan,eps1=np.nan,plane_used=0,eps2=np.nan,inliers=0,plane_rmse=np.nan))

    for mid in range(K):
        if mid in sky_masks:
            continue
        pts_m = df[df['mask_id'] == mid]
        if len(pts_m) < 2:
            continue

        r_arr = pts_m['r'].values.astype(np.float32)
        z_arr = pts_m['Z'].values.astype(np.float32)
        w_arr = pts_m['omega'].values.astype(np.float32)
        ok = np.isfinite(r_arr) & np.isfinite(z_arr) & np.isfinite(w_arr) & (w_arr > 0)
        r_arr, z_arr, w_arr = r_arr[ok], z_arr[ok], w_arr[ok]
        if r_arr.size < 2:
            continue

        s, t, ok_fit = fit_affine_weighted(r_arr, z_arr, w_arr)
        if not ok_fit or not np.isfinite(s) or not np.isfinite(t):
            continue
        s_dict[mid], t_dict[mid] = float(s), float(t)

        pred_affine_pts = s * r_arr + t
        eps1 = weighted_rmse(pred_affine_pts, z_arr, w_arr)

        depth_affine[bitmaps[mid]] = (s * r[bitmaps[mid]] + t).astype(np.float32)

        for rec in logs:
            if rec['frame']==frame and rec['mask_id']==mid:
                rec.update(s=float(s), t=float(t), eps1=float(eps1))
                break

    if use_plane:
        for mid in list(s_dict.keys()):
            if mid in sky_masks:
                continue
            pts_m = df[df['mask_id'] == mid]
            if len(pts_m) < 3:
                continue
            pts3d = pts_m[['X','Y','Z']].values.astype(np.float32)
            plane, inlier_mask, rmse = ransac_plane(pts3d, n_iters=plane_iters, thresh=plane_thresh)
            if plane is None:
                continue

            n_inl = int(inlier_mask.sum())
            inlier_ratio = n_inl / max(1, len(pts3d))
            rmse = float(rmse) if rmse is not None else 1e6

            w_plane = inlier_ratio * np.exp(-rmse / max(plane_thresh, 1e-6))
            w_plane = float(np.clip(w_plane, 0.0, 1.0))

            # 查找当前mask的eps1
            eps1 = np.inf
            for rec in logs:
                if rec['frame']==frame and rec['mask_id']==mid:
                    eps1 = rec['eps1']
                    break
            if not np.isfinite(eps1):
                continue

            s, t = s_dict[mid], t_dict[mid]
            pred_affine_pts = s * pts_m['r'].values.astype(np.float32) + t
            z_plane_map = plane_depth_map(plane, fx, fy, cx, cy, H, W, d_max=d_max)
            pred_plane_pts, ok2 = sample_at_pixels(z_plane_map, pts_m['px'].values, pts_m['py'].values)
            pred_mix_pts = (1.0 - w_plane) * pred_affine_pts + w_plane * pred_plane_pts
            eps2 = weighted_rmse(pred_mix_pts,
                                 pts_m['Z'].values.astype(np.float32),
                                 pts_m['omega'].values.astype(np.float32))

            improved = (np.isfinite(eps1) and np.isfinite(eps2) and (eps2 < eps1*(1.0 - plane_improve_ratio)))

            # 更新日志
            for rec in logs:
                if rec['frame']==frame and rec['mask_id']==mid:
                    rec.update(plane_used=int(bool(improved)), eps2=float(eps2),
                               inliers=int(n_inl), plane_rmse=float(rmse))
                    break

            if improved:
                used_plane_masks.add(mid)
                d_aff = (s * r + t).astype(np.float32)
                d_mix = (1.0 - w_plane) * d_aff + w_plane * z_plane_map
                depth_final[bitmaps[mid]] = d_mix[bitmaps[mid]].astype(np.float32)

    # 填充未优化的mask深度
    for mid in s_dict.keys():
        if mid in used_plane_masks or mid in sky_masks:
            continue
        depth_final[bitmaps[mid]] = depth_affine[bitmaps[mid]]

    # 填充缺失深度的像素
    nan_pix = ~np.isfinite(depth_final)
    if np.any(nan_pix):
        valid_mids = [m for m in s_dict.keys() if m not in sky_masks]
        centers = per_mask_centers(bitmaps, valid_mids)
        s_map, t_map = nearest_mask_param_fill(H, W, centers,
                                               {m: s_dict[m] for m in centers.keys()},
                                               {m: t_dict[m] for m in centers.keys()})
        if not np.any(np.isfinite(s_map)):
            # 全局拟合兜底
            if len(df) >= 2:
                s_g, t_g, ok_fit = fit_affine_weighted(df['r'].values.astype(np.float32),
                                                       df['Z'].values.astype(np.float32),
                                                       df['omega'].values.astype(np.float32))
                if ok_fit and np.isfinite(s_g) and np.isfinite(t_g):
                    depth_final[nan_pix] = (s_g * r[nan_pix] + t_g).astype(np.float32)
                else:
                    depth_final[nan_pix] = d_max
            else:
                depth_final[nan_pix] = d_max
        else:
            depth_final[nan_pix] = (s_map[nan_pix] * r[nan_pix] + t_map[nan_pix]).astype(np.float32)

    # 最终深度后处理
    depth_final[~np.isfinite(depth_final)] = d_max
    depth_final = np.clip(depth_final, 0.0, d_max).astype(np.float32)
    write_depth_outputs(out_dir, frame, depth_final, d_max=d_max)
    return pd.DataFrame(logs)