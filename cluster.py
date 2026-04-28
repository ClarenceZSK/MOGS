import os
import colorsys
import numpy as np
import cv2
import pandas as pd

def safe_round_xy(x, y, W, H):
    ix = int(round(x)); iy = int(round(y))
    if ix < 0: ix = 0
    if iy < 0: iy = 0
    if ix >= W: ix = W - 1
    if iy >= H: iy = H - 1
    return ix, iy

def load_depth_png_for_cluster(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取深度图：{path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        depth = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        depth = img.astype(np.float32) / 255.0
    else:
        depth = img.astype(np.float32)
    return depth

def build_instance_map_depth_aware(bitmaps: np.ndarray, depth: np.ndarray,
                                   smaller_is_closer: bool = False):
    N, H, W = bitmaps.shape
    rep_depth = np.full(N, np.inf, np.float32)
    for i in range(N):
        m = bitmaps[i].astype(bool)
        if m.any():
            dvals = depth[m]
            dvals = dvals[np.isfinite(dvals)]
            if dvals.size > 0:
                rep_depth[i] = np.median(dvals)
    order = np.argsort(rep_depth)
    if not smaller_is_closer:
        order = order[::-1]
    inst = np.full((H, W), -1, dtype=np.int32)
    for i in order:
        m = bitmaps[i].astype(bool)
        inst[m] = int(i)
    return inst, rep_depth

def neighbor_vote(inst_map: np.ndarray, x: int, y: int, r: int, exclude_id=None):
    H, W = inst_map.shape
    y0 = max(0, y - r); y1 = min(H, y + r + 1)
    x0 = max(0, x - r); x1 = min(W, x + r + 1)
    win = inst_map[y0:y1, x0:x1]
    vals = win.reshape(-1)
    vals = vals[vals >= 0]
    if exclude_id is not None:
        vals = vals[vals != exclude_id]
    if vals.size == 0:
        return -1
    uniq, cnt = np.unique(vals, return_counts=True)
    return int(uniq[np.argmax(cnt)])

def _relabel_disconnected_components(inst_map: np.ndarray, keep_largest: bool = True) -> np.ndarray:
    out = inst_map.copy()
    present = np.unique(out[out >= 0])
    if present.size == 0:
        return out
    next_id = int(present.max()) + 1
    for mid in present:
        m = (out == mid).astype(np.uint8)
        if not m.any():
            continue
        nlab, labels = cv2.connectedComponents(m, connectivity=4)
        if nlab <= 2:
            continue
        comp_sizes = np.bincount(labels.reshape(-1))[1:]
        order = np.argsort(comp_sizes)[::-1]
        if keep_largest:
            keep_comp = 1 + int(order[0])
            for comp_idx in range(1, nlab):
                if comp_idx == keep_comp:
                    continue
                ys, xs = np.where(labels == comp_idx)
                if ys.size == 0:
                    continue
                out[ys, xs] = next_id
                next_id += 1
        else:
            first = 1 + int(order[0])
            for comp_idx in range(1, nlab):
                if comp_idx == first:
                    continue
                ys, xs = np.where(labels == comp_idx)
                if ys.size == 0:
                    continue
                out[ys, xs] = next_id
                next_id += 1
    return out

def _merge_small_masks_by_area(inst_map: np.ndarray, min_pixels: int, cand_win_radius: int, max_mean_dist: float):
    H, W = inst_map.shape
    out = inst_map.copy()
    merged_any = False
    merged_count = 0
    ids = np.unique(out[out >= 0])
    if ids.size == 0:
        return out, merged_any, merged_count
    areas = {int(i): int((out == i).sum()) for i in ids}
    small_ids = [i for i in ids if areas[int(i)] < min_pixels]
    small_ids = sorted(small_ids, key=lambda k: areas[int(k)])
    for sid in small_ids:
        m = (out == sid)
        if not m.any():
            continue
        ys, xs = np.where(m)
        y0 = max(0, ys.min() - cand_win_radius)
        y1 = min(H, ys.max() + cand_win_radius + 1)
        x0 = max(0, xs.min() - cand_win_radius)
        x1 = min(W, xs.max() + cand_win_radius + 1)
        win = out[y0:y1, x0:x1]
        frag = (win == sid).astype(np.uint8)
        touch_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        frag_touch = cv2.dilate(frag, touch_kernel).astype(bool)
        cand_ids = np.unique(win[frag_touch])
        cand_ids = [int(c) for c in cand_ids if c >= 0 and c != sid]
        if not cand_ids:
            continue
        ys_rel = ys - y0; xs_rel = xs - x0
        best_id, best_mean_d = None, None
        for cid in cand_ids:
            cm = (win == cid).astype(np.uint8)
            dt = cv2.distanceTransform((cm == 0).astype(np.uint8), cv2.DIST_L2, 3)
            md = float(dt[ys_rel, xs_rel].mean())
            if (best_mean_d is None) or (md < best_mean_d):
                best_mean_d, best_id = md, cid
        if best_id is None or best_mean_d is None:
            continue
        if best_mean_d > max_mean_dist:
            continue
        out[m] = best_id
        merged_any = True
        merged_count += 1
    return out, merged_any, merged_count

def _reindex_inst_map(inst_map: np.ndarray):
    out = inst_map.copy()
    ids = sorted(np.unique(out[out >= 0]).tolist())
    mapping = {old: new for new, old in enumerate(ids)}
    for old, new in mapping.items():
        out[out == old] = new
    return out, mapping

def _rep_depth_from_inst_map(inst_map: np.ndarray, depth: np.ndarray) -> np.ndarray:
    ids = sorted(np.unique(inst_map[inst_map >= 0]).tolist())
    K = len(ids)
    rep = np.full(K, np.inf, np.float32)
    for i in range(K):
        m = (inst_map == i)
        if not m.any():
            continue
        dvals = depth[m]
        dvals = dvals[np.isfinite(dvals)]
        if dvals.size:
            rep[i] = np.median(dvals)
    return rep

def _reassign_to_nearby_closer_mask(ix, iy, mid0, inst_map, rep_depth, r):
    H, W = inst_map.shape
    y0 = max(0, iy - r); y1 = min(H, iy + r + 1)
    x0 = max(0, ix - r); x1 = min(W, ix + r + 1)
    win = inst_map[y0:y1, x0:x1]
    vals = win.reshape(-1)
    vals = vals[vals >= 0]
    if vals.size == 0:
        return mid0
    cand_ids = np.unique(vals)
    if mid0 >= 0:
        cand_ids = cand_ids[cand_ids != mid0]
    if cand_ids.size == 0:
        return mid0
    best_id, best_d2 = None, None
    for cid in cand_ids:
        ys, xs = np.where(win == cid)
        if ys.size == 0:
            continue
        ys_full = ys + y0; xs_full = xs + x0
        dy = ys_full - iy; dx = xs_full - ix
        d2 = (dx * dx + dy * dy).min()
        if (best_d2 is None) or (d2 < best_d2):
            best_d2, best_id = d2, int(cid)
    if best_id is None:
        return mid0
    d_orig = rep_depth[mid0] if (mid0 >= 0 and mid0 < rep_depth.size) else np.inf
    d_cand = rep_depth[best_id] if (best_id < rep_depth.size) else np.inf
    return best_id if (d_cand > d_orig) else mid0

def _find_nearest_mask_excluding(ix, iy, inst_map, exclude_ids, r_init, r_max):
    H, W = inst_map.shape
    r = max(1, int(r_init))
    r_max = max(r, int(r_max))
    best_id, best_d2 = -1, None
    while r <= r_max:
        y0 = max(0, iy - r); y1 = min(H, iy + r + 1)
        x0 = max(0, ix - r); x1 = min(W, ix + r + 1)
        win = inst_map[y0:y1, x0:x1]
        vals = win.reshape(-1)
        vals = vals[vals >= 0]
        if vals.size:
            cand_ids = np.unique(vals)
            if exclude_ids:
                cand_ids = [cid for cid in cand_ids if cid not in exclude_ids]
            if cand_ids:
                for cid in cand_ids:
                    ys, xs = np.where(win == cid)
                    if ys.size == 0:
                        continue
                    ys_full = ys + y0; xs_full = xs + x0
                    dy = ys_full - iy; dx = xs_full - ix
                    d2 = (dx * dx + dy * dy).min()
                    if (best_d2 is None) or (d2 < best_d2):
                        best_d2, best_id = d2, int(cid)
                if best_id != -1:
                    return best_id
        r += max(1, r // 2)
    return -1

def _adjacency_from_inst(inst_map: np.ndarray, ids: list[int]):
    H, W = inst_map.shape
    adj = {i: set() for i in ids}
    a = inst_map[:, :-1]; b = inst_map[:, 1:]
    mask = (a != b) & (a >= 0) & (b >= 0)
    for y, x in zip(*np.where(mask)):
        i, j = int(a[y, x]), int(b[y, x])
        if i != j: adj[i].add(j); adj[j].add(i)
    a = inst_map[:-1, :]; b = inst_map[1:, :]
    mask = (a != b) & (a >= 0) & (b >= 0)
    for y, x in zip(*np.where(mask)):
        i, j = int(a[y, x]), int(b[y, x])
        if i != j: adj[i].add(j); adj[j].add(i)
    return adj

def _greedy_color(ids: list[int], adj: dict[int, set], K_palette: int = 24):
    palette = []
    for k in range(K_palette):
        h = k / K_palette
        s, v = 0.8, 0.95
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        palette.append((int(b*255), int(g*255), int(r*255)))
    order = sorted(ids, key=lambda i: len(adj.get(i, [])), reverse=True)
    color_of = {}
    for i in order:
        forbidden = {color_of[j] for j in adj.get(i, []) if j in color_of}
        for cidx in range(K_palette):
            if cidx not in forbidden:
                color_of[i] = cidx
                break
        if i not in color_of:
            color_of[i] = 0
    id2bgr = {i: palette[color_of[i]] for i in ids}
    return id2bgr

def process_clustering_for_frame(frame_id, depth_png_path, mask_npz_path, points_csv_path,
                                 output_mask_dir, output_csv_dir, output_viz_dir, 
                                 output_point_viz_dir, image_folder):
    """
    对单帧进行聚类优化处理，并生成两种可视化
    """
    # 读取深度图
    depth = load_depth_png_for_cluster(depth_png_path)
    H, W = depth.shape

    # 读取原始掩码
    data = np.load(mask_npz_path)
    if 'bitmaps' in data:
        bitmaps_raw = data['bitmaps']
    else:
        bitmaps_raw = data['arr_0']
    bitmaps = (bitmaps_raw > 0).astype(np.uint8)
    if bitmaps.ndim == 2:
        bitmaps = bitmaps[np.newaxis, ...]
    N, Hm, Wm = bitmaps.shape
    if (Hm, Wm) != (H, W):
        raise ValueError(f"深度图尺寸 ({H},{W}) 与掩码尺寸 ({Hm},{Wm}) 不匹配")

    # 构建深度感知的实例映射图
    inst_map, rep_depth = build_instance_map_depth_aware(bitmaps, depth, smaller_is_closer=False)
    inst_map = _relabel_disconnected_components(inst_map, keep_largest=True)
    
    # 合并小掩码
    SMALL_MASK_MIN_PIXELS = 300
    SMALL_MASK_CAND_WIN_RADIUS = 12
    SMALL_MASK_MAX_MEAN_DIST = 4.5
    inst_map, merged_any, merged_count = _merge_small_masks_by_area(
        inst_map, SMALL_MASK_MIN_PIXELS, SMALL_MASK_CAND_WIN_RADIUS, SMALL_MASK_MAX_MEAN_DIST
    )
    
    # 重新索引
    inst_map, _ = _reindex_inst_map(inst_map)
    rep_depth = _rep_depth_from_inst_map(inst_map, depth)
    K = int(np.max(inst_map)) + 1 if np.any(inst_map >= 0) else 0

    # 读取特征点
    df_points = pd.read_csv(points_csv_path)
    required_cols = ['point_id', 'pixel_x', 'pixel_y', 'X', 'Y', 'Z', 'omega']
    for col in required_cols:
        if col not in df_points.columns:
            raise ValueError(f"特征点CSV缺少列: {col}")

    # 分配 mask_id
    mask_ids = []
    xs = df_points['pixel_x'].to_numpy()
    ys = df_points['pixel_y'].to_numpy()
    for x, y in zip(xs, ys):
        ix, iy = safe_round_xy(x, y, W, H)
        mid = int(inst_map[iy, ix])
        if mid < 0:
            mid = neighbor_vote(inst_map, ix, iy, 2, exclude_id=None)
        mask_ids.append(mid)
    df_points['mask_id'] = mask_ids

    # 边缘细化
    EDGE_REFINE_ENABLED = True
    EDGE_REFINE_RADIUS = 3
    if EDGE_REFINE_ENABLED and EDGE_REFINE_RADIUS > 0:
        new_mask_ids = df_points['mask_id'].to_numpy()
        for i_row, (mid, x, y) in enumerate(zip(df_points['mask_id'], xs, ys)):
            ix, iy = safe_round_xy(x, y, W, H)
            new_mid = _reassign_to_nearby_closer_mask(ix, iy, int(mid), inst_map, rep_depth, r=EDGE_REFINE_RADIUS)
            new_mask_ids[i_row] = new_mid
        df_points['mask_id'] = new_mask_ids

    # 零深度掩码重分配
    ZERO_DEPTH_REASSIGN_ENABLED = True
    ZERO_DEPTH_EPS = 1e-8
    ZERO_DEPTH_INIT_RADIUS = 3
    ZERO_DEPTH_MAX_RADIUS = 25
    if ZERO_DEPTH_REASSIGN_ENABLED:
        bad_mask_ids = set(np.where(np.isfinite(rep_depth) & (np.abs(rep_depth) <= ZERO_DEPTH_EPS))[0].tolist())
        if bad_mask_ids:
            new_mask_ids = df_points['mask_id'].to_numpy()
            for i_row, (mid, x, y) in enumerate(zip(df_points['mask_id'], xs, ys)):
                if int(mid) in bad_mask_ids:
                    ix, iy = safe_round_xy(x, y, W, H)
                    best_id = _find_nearest_mask_excluding(
                        ix, iy, inst_map, exclude_ids=bad_mask_ids,
                        r_init=ZERO_DEPTH_INIT_RADIUS, r_max=ZERO_DEPTH_MAX_RADIUS
                    )
                    new_mask_ids[i_row] = best_id if best_id != -1 else -1
            df_points['mask_id'] = new_mask_ids

    # 保存优化后的掩码
    os.makedirs(output_mask_dir, exist_ok=True)
    mask_out_path = os.path.join(output_mask_dir, f"{frame_id:06d}_bitmaps.npz")
    if K > 0:
        bitmaps_out = (inst_map[None, :, :] == np.arange(K)[:, None, None]).astype(np.uint8)
        np.savez_compressed(mask_out_path, bitmaps=bitmaps_out)
    else:
        np.savez_compressed(mask_out_path, bitmaps=np.array([]))

    # 保存CSV
    os.makedirs(output_csv_dir, exist_ok=True)
    csv_out_path = os.path.join(output_csv_dir, f"cluster_by_mask_{frame_id:06d}.csv")
    df_out = df_points.copy()
    if 'frame_id' in df_out.columns:
        df_out = df_out.drop(columns=['frame_id'])
    df_out.insert(0, 'frame_id', frame_id)
    target_cols = ['frame_id', 'mask_id', 'point_id', 'pixel_x', 'pixel_y', 'X', 'Y', 'Z', 'omega']
    for col in target_cols:
        if col not in df_out.columns:
            df_out[col] = -1
    df_out = df_out[target_cols]
    df_out.to_csv(csv_out_path, index=False)

    return mask_out_path, csv_out_path