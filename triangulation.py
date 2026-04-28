import numpy as np
from scipy.optimize import least_squares

# 相机参数（KITTI 00）
K = np.array([
    [7.188560000000e+02, 0, 6.071928000000e+02],
    [0, 7.188560000000e+02, 1.852157000000e+02],
    [0, 0, 1]
], dtype=np.float64)

def load_poses(pose_path: str):
    """加载 KITTI 位姿文件，返回世界→相机的旋转列表和平移列表"""
    Rs, Ts = [], []
    with open(pose_path, 'r') as f:
        for line in f:
            data = list(map(float, line.strip().split()))
            mat = np.array(data, dtype=np.float64).reshape(3, 4)
            R_c2w = mat[:, :3]
            t_c2w = mat[:, 3]
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w
            Rs.append(R_w2c)
            Ts.append(t_w2c)
    return Rs, Ts

def linear_triangulate(obs, Ps):
    """线性三角化（SVD）"""
    A = []
    for (u, v), P in zip(obs, Ps):
        A.append(u * P[2, :] - P[0, :])
        A.append(v * P[2, :] - P[1, :])
    A = np.stack(A)
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    return X_h[:3] / X_h[3]

def reprojection_residuals_rt(X, obs, R_list, t_list, K):
    """重投影残差（用于非线性优化）"""
    X = np.asarray(X, dtype=np.float64).reshape(3)
    res = np.empty(2 * len(obs), dtype=np.float64)
    kx = 0
    for (u, v), R, t in zip(obs, R_list, t_list):
        Xc = R @ X + t
        xc, yc, zc = Xc
        if zc <= 1e-8:
            res[kx] = 1e3
            res[kx+1] = 1e3
        else:
            u_hat = K[0,0] * (xc / zc) + K[0,2]
            v_hat = K[1,1] * (yc / zc) + K[1,2]
            res[kx] = u_hat - u
            res[kx+1] = v_hat - v
        kx += 2
    return res

def optimise_point(X0, obs, Ps, K):
    """非线性优化 3D 点（最小化重投影误差）"""
    X0 = np.asarray(X0, dtype=np.float64).reshape(3)
    K_inv = np.linalg.inv(K)
    Rt_list = [K_inv @ P for P in Ps]
    R_list = [Rt[:, :3] for Rt in Rt_list]
    t_list = [Rt[:, 3] for Rt in Rt_list]

    def jac(X):
        X = np.asarray(X, dtype=np.float64).reshape(3)
        J = np.zeros((2 * len(obs), 3), dtype=np.float64)
        for i, (R, t) in enumerate(zip(R_list, t_list)):
            Xc = R @ X + t
            xc, yc, zc = Xc
            if zc <= 1e-8:
                continue
            r1, r2, r3 = R[0,:], R[1,:], R[2,:]
            Ju = (K[0,0] / (zc * zc)) * (r1 * zc - xc * r3)
            Jv = (K[1,1] / (zc * zc)) * (r2 * zc - yc * r3)
            J[2*i,   :] = Ju
            J[2*i+1, :] = Jv
        return J

    res = least_squares(
        lambda X: reprojection_residuals_rt(X, obs, R_list, t_list, K),
        X0, jac=jac, method='trf', loss='huber',
        f_scale=1.0, max_nfev=100, x_scale='jac', verbose=0
    )
    return res.x

def compute_confidence(obs_list, R_list, t_list, X_opt, K, tau=2.0, thr=3.0):
    """计算置信度 omega（基于内点比例和平均重投影误差）"""
    X = np.asarray(X_opt, dtype=float).reshape(3, 1)
    errs, inliers = [], 0
    for (u, v), R, t in zip(obs_list, R_list, t_list):
        Xc = R @ X + t.reshape(3, 1)
        zc = float(Xc[2, 0])
        if zc <= 1e-8 or not np.isfinite(zc):
            e = 1e3
        else:
            u_hat = K[0,0] * (Xc[0,0] / zc) + K[0,2]
            v_hat = K[1,1] * (Xc[1,0] / zc) + K[1,2]
            e = float(np.hypot(u_hat - u, v_hat - v))
        errs.append(e)
        if e <= thr:
            inliers += 1
    if not errs:
        return 0.0
    e_mean = float(np.mean(errs))
    inlier_ratio = inliers / len(errs)
    omega = inlier_ratio * np.exp(- (e_mean / tau) ** 2)
    return float(np.clip(omega, 0.0, 1.0))