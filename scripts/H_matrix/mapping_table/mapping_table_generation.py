#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

# ------------------------- 基础常量 ------------------------- #
IMG_FILES: Sequence[str] = [
    #example:
   # "cam0.png",
   # "cam1.png",
   # "cam2.png",
   # "cam3.png",
   # "cam4.png",
   # "cam5.png",
   # "cam6.png",
   # "cam7.png",
]

FOCAL_LENGTH = 2926.82
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_sift_mapping")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 预处理 & 匹配配置
CROP_RATIO = 0.02
MATCH_SCALE = 0.7
RATIO_TEST = 0.8
CLAHE_CLIP = 2.0
CLAHE_GRID = 8
MIN_INLIERS = 30
CONFIDENCE_THRESH = 0.02


# 映射表常量
MAPPING_TABLE_VERSION = 2
MAPPING_TABLE_RESERVED_BYTES = 10
BYTES_PER_KB = 1024
FLOAT_PRECISION = 4
TARGET_CANVAS_HEIGHT = 2160  # 若最终画布高于该值，则上下裁剪为指定高度


# ------------------------- 数据结构 ------------------------- #
@dataclass
class FeatureImage:
    color: np.ndarray
    gray_scaled: np.ndarray
    scale: float
    crop_offset: int


# ------------------------- 柱面映射 ------------------------- #
def build_inverse_cylindrical_maps(width: int, height: int, focal: float):
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    xs, ys = np.meshgrid(xs, ys)
    x_c = xs - width / 2.0
    y_c = ys - height / 2.0
    theta = x_c / focal
    h_ = y_c / focal
    X = np.sin(theta)
    Y = h_
    Z = np.cos(theta)
    map_x = focal * X / Z + width / 2.0
    map_y = focal * Y / Z + height / 2.0
    return map_x.astype(np.float32), map_y.astype(np.float32)


def build_forward_cylindrical_maps(width: int, height: int, focal: float):
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    xs, ys = np.meshgrid(xs, ys)
    x_c = xs - width / 2.0
    y_c = ys - height / 2.0
    theta = np.arctan2(x_c, focal)
    denom = np.sqrt(x_c**2 + focal**2)
    h_ = y_c / denom
    cyl_x = focal * theta + width / 2.0
    cyl_y = focal * h_ + height / 2.0
    return cyl_x.astype(np.float32), cyl_y.astype(np.float32)


def cylindrical_project_image(img: np.ndarray, inv_map_x: np.ndarray, inv_map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        img,
        inv_map_x,
        inv_map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ------------------------- 预处理 & 匹配 ------------------------- #
def preprocess_image(img: np.ndarray, crop_ratio=CROP_RATIO, match_scale=MATCH_SCALE) -> FeatureImage:
    if img is None:
        return None
    h, w = img.shape[:2]
    crop_px = int(w * crop_ratio)
    crop_px = min(crop_px, w // 4)
    cropped = img[:, crop_px : w - crop_px].copy() if crop_px > 0 else img.copy()

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_GRID, CLAHE_GRID))
    gray_eq = clahe.apply(gray)
    if abs(match_scale - 1.0) > 1e-6:
        gray_scaled = cv2.resize(gray_eq, None, fx=match_scale, fy=match_scale, interpolation=cv2.INTER_AREA)
    else:
        gray_scaled = gray_eq
    return FeatureImage(color=img, gray_scaled=gray_scaled, scale=match_scale, crop_offset=crop_px)


def rescale_point(kp, scale: float, crop_offset: int):
    return [kp.pt[0] / scale + crop_offset, kp.pt[1] / scale]


def sift_match_and_translation(feat_a: FeatureImage,
                               feat_b: FeatureImage,
                               confidence_thresh=CONFIDENCE_THRESH,
                               ratio_thresh=RATIO_TEST):
    sift = cv2.SIFT_create()
    kp_a, des_a = sift.detectAndCompute(feat_a.gray_scaled, None)
    kp_b, des_b = sift.detectAndCompute(feat_b.gray_scaled, None)
    if des_a is None or des_b is None:
        raise RuntimeError("SIFT 未检测到足够特征点")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in knn if m.distance < ratio_thresh * n.distance]

    denom = max(1, min(len(kp_a), len(kp_b)))
    confidence = len(good) / denom
    if confidence < confidence_thresh:
        raise RuntimeError(f"匹配置信度 {confidence:.4f} < {confidence_thresh}")

    pts_a = np.float32([rescale_point(kp_a[m.queryIdx], feat_a.scale, feat_a.crop_offset) for m in good])
    pts_b = np.float32([rescale_point(kp_b[m.trainIdx], feat_b.scale, feat_b.crop_offset) for m in good])

    H, mask = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 4.0)
    if H is None or mask is None:
        raise RuntimeError("RANSAC 未能估计单应性矩阵")
    inliers = int(mask.sum())
    if inliers < MIN_INLIERS:
        raise RuntimeError(f"RANSAC 内点数 {inliers} < {MIN_INLIERS}")

    deltas = pts_a[mask.ravel() == 1] - pts_b[mask.ravel() == 1]
    tx = float(np.median(deltas[:, 0]))
    ty = float(np.median(deltas[:, 1]))
    translation = np.array([[1.0, 0.0, tx],
                            [0.0, 1.0, ty],
                            [0.0, 0.0, 1.0]], dtype=np.float32)
    print(
        f"匹配置信度 {confidence:.4f}，内点 {inliers}/{len(good)}，平移 (tx={tx:.2f}, ty={ty:.2f})"
    )
    return translation


def compute_transforms(features: Sequence[FeatureImage]) -> List[np.ndarray]:
    transforms = [np.eye(3, dtype=np.float32)]
    for i in range(len(features) - 1):
        print(f"\n估计 cam{i} -> cam{i+1} 平移...")
        T = sift_match_and_translation(features[i], features[i + 1])
        transforms.append(transforms[-1] @ T)
    return transforms


# ------------------------- 缝合与画布 ------------------------- #
def compute_canvas_bounds(images: Sequence[np.ndarray], transforms: Sequence[np.ndarray]):
    corners_all = []
    for img, H in zip(images, transforms):
        h, w = img.shape[:2]
        pts = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float32).T
        warped = H @ pts
        warped /= warped[2]
        corners_all.append(warped[:2])
    corners_all = np.hstack(corners_all)
    x_min = int(np.floor(corners_all[0].min()))
    x_max = int(np.ceil(corners_all[0].max()))
    y_min = int(np.floor(corners_all[1].min()))
    y_max = int(np.ceil(corners_all[1].max()))
    shift_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    print(f"画布尺寸: {canvas_w}x{canvas_h}")
    return shift_mat, canvas_w, canvas_h


def determine_vertical_crop(canvas_h: int, target_h: int):
    if target_h is None or target_h <= 0 or target_h >= canvas_h:
        return 0, canvas_h
    delta = canvas_h - target_h
    top_trim = delta
    bottom_trim = 0
    start = max(0, top_trim)
    end = min(canvas_h, canvas_h - bottom_trim)
    return start, end


def compute_min_energy_seam(diff_region: np.ndarray):
    energy = diff_region.astype(np.float32).mean(axis=2)
    h, w = energy.shape
    dp = energy.copy()
    backtrack = np.zeros((h, w), dtype=np.int32)
    for y in range(1, h):
        for x in range(w):
            candidates = [x]
            if x > 0:
                candidates.append(x - 1)
            if x + 1 < w:
                candidates.append(x + 1)
            best = min(candidates, key=lambda c: dp[y - 1, c])
            dp[y, x] += dp[y - 1, best]
            backtrack[y, x] = best
    seam = []
    col = int(np.argmin(dp[-1]))
    for y in reversed(range(h)):
        seam.append((y, col))
        col = backtrack[y, col]
    seam.reverse()
    return seam


def blend_with_seam(base, base_mask, warped, warped_mask, seam_info):
    y_min, y_max, x_min, x_max = seam_info["overlap_bbox"]
    seam_lookup = seam_info["seam_x_lookup"]
    result = base.copy()
    for y in range(y_min, y_max):
        seam_x = seam_lookup.get(y, x_min)
        seam_x = int(np.clip(seam_x, x_min, x_max))
        result[y, x_min:seam_x] = base[y, x_min:seam_x]
        result[y, seam_x:x_max] = warped[y, seam_x:x_max]
    non_overlap = (base_mask == 0) & (warped_mask > 0)
    result[non_overlap] = warped[non_overlap]
    new_mask = np.where(warped_mask > 0, 255, base_mask)
    return result, new_mask





def warp_all_images(images, transforms, shift_mat, canvas_w, canvas_h):
    warped_images = []
    warped_masks = []
    warp_mats = []
    for idx, (img, H) in enumerate(zip(images, transforms)):
        warp_mat = shift_mat @ H
        warped = cv2.warpPerspective(img, warp_mat, (canvas_w, canvas_h))
        mask = cv2.warpPerspective(
            np.ones(img.shape[:2], dtype=np.uint8) * 255,
            warp_mat,
            (canvas_w, canvas_h),
            flags=cv2.INTER_NEAREST,
        )
        warped_images.append(warped)
        warped_masks.append(mask)
        warp_mats.append(warp_mat)
       
    return warped_images, warped_masks, warp_mats


def build_seam_info(warped_images, warped_masks):
    panorama = warped_images[0].copy()
    mask = warped_masks[0].copy()
    seam_info_list = []
    for idx in range(1, len(warped_images)):
        warped = warped_images[idx]
        w_mask = warped_masks[idx]
        overlap = (mask > 0) & (w_mask > 0)
        if not np.any(overlap):
            panorama[w_mask > 0] = warped[w_mask > 0]
            mask = np.where(w_mask > 0, 255, mask)
            continue
        ys, xs = np.where(overlap)
        y_min, y_max = ys.min(), ys.max() + 1
        x_min, x_max = xs.min(), xs.max() + 1
        base_region = panorama[y_min:y_max, x_min:x_max]
        new_region = warped[y_min:y_max, x_min:x_max]
        diff = cv2.absdiff(base_region, new_region)
        seam = compute_min_energy_seam(diff)
        seam_lookup = {}
        for (row_rel, col_rel) in seam:
            seam_lookup[y_min + row_rel] = x_min + col_rel
        
        # 计算缝合线的最左边和最右边像素
        seam_x_values = list(seam_lookup.values())
        if seam_x_values:
            seam_x_min = min(seam_x_values)
            seam_x_max = max(seam_x_values)
            print(f"缝合缝 {idx-1}->{idx}: 最左边={seam_x_min}, 最右边={seam_x_max}")
        else:
            print(f"缝合缝 {idx-1}->{idx}: 无有效缝合线")
        
        info = {
            "cam_id1": idx - 1,
            "cam_id2": idx,
            "overlap_bbox": (y_min, y_max, x_min, x_max),
            "seam_x_lookup": seam_lookup,
        }
        seam_info_list.append(info)
        panorama, mask = blend_with_seam(panorama, mask, warped, w_mask, info)
        print(f"完成 cam{idx-1} -> cam{idx} seam。")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "panorama_sift_dp.jpg"), panorama)
    return seam_info_list, panorama


# ------------------------- 映射表生成 ------------------------- #
def generate_mapping_dict(forward_map_x, forward_map_y, warp_mats, seam_info_dict, canvas_w, canvas_h):
    mapping: Dict[Tuple[int, int, int], Tuple[int, int]] = {}
    height, width = forward_map_x.shape
    for cam_id, warp_mat in enumerate(warp_mats):
        print(f"\n生成 cam{cam_id} 映射...")
        seam_info = seam_info_dict.get(cam_id)
        seam_map = None
        bbox = None
        if seam_info:
            seam_map = np.full(canvas_h, -1, dtype=np.int32)
            for row, seam_x in seam_info["seam_x_lookup"].items():
                if 0 <= row < canvas_h:
                    seam_map[row] = seam_x
            bbox = seam_info["overlap_bbox"]

        for src_y in range(height):
            pts = np.vstack(
                (
                    forward_map_x[src_y],
                    forward_map_y[src_y],
                    np.ones(width, dtype=np.float32),
                )
            )
            canvas_pts = warp_mat @ pts
            denom = canvas_pts[2]
            valid = denom != 0
            x_canvas = np.round(canvas_pts[0] / denom).astype(np.int32)
            y_canvas = np.round(canvas_pts[1] / denom).astype(np.int32)
            valid &= (x_canvas >= 0) & (x_canvas < canvas_w) & (y_canvas >= 0) & (y_canvas < canvas_h)
            cols = np.where(valid)[0]
            for src_x in cols:
                out_x = int(x_canvas[src_x])
                out_y = int(y_canvas[src_x])
                if seam_info and bbox is not None:
                    y_min, y_max, x_min, x_max = bbox
                    if y_min <= out_y < y_max and x_min <= out_x < x_max:
                        seam_x = seam_map[out_y]
                        if seam_x >= 0 and out_x < seam_x:
                            continue
                mapping[(cam_id, src_x, src_y)] = (out_x, out_y)
        mapped_pixels = sum(1 for key in mapping if key[0] == cam_id)
        print(f"  cam{cam_id} 有效像素: {mapped_pixels}")
    return mapping


def crop_mapping_vertically(mapping_dict, crop_start, crop_end, canvas_h):
    """裁剪映射表的垂直范围。"""
    if crop_start <= 0 and crop_end >= canvas_h:
        return mapping_dict, canvas_h
    cropped = {}
    for key, (out_x, out_y) in mapping_dict.items():
        if crop_start <= out_y < crop_end:
            cropped[key] = (out_x, out_y - crop_start)
    new_height = crop_end - crop_start
    print(f"垂直裁剪映射: 保留 {new_height} 行, 覆盖条目 {len(cropped)}")
    return cropped, new_height


def save_mapping_table_binary(mapping_dict, canvas_w, canvas_h,
                              single_width, single_height, cam_num, output_path):
    print(f"\n保存映射表: {output_path}")
    with open(output_path, "wb") as f:
        reserved = bytes(MAPPING_TABLE_RESERVED_BYTES)
        f.write(struct.pack("<H", MAPPING_TABLE_VERSION))
        f.write(struct.pack("<I", canvas_w))
        f.write(struct.pack("<I", canvas_h))
        f.write(struct.pack("<H", cam_num))
        f.write(struct.pack("<I", single_width))
        f.write(struct.pack("<I", single_height))
        f.write(reserved)
        num_mappings = len(mapping_dict)
        f.write(struct.pack("<I", num_mappings))
        sorted_items = sorted(mapping_dict.items(), key=lambda x: (x[0][0], x[0][2], x[0][1]))
        for (cam_id, src_x, src_y), (out_x, out_y) in sorted_items:
            f.write(struct.pack("<H", cam_id))
            f.write(struct.pack("<H", src_x))
            f.write(struct.pack("<H", src_y))
            f.write(struct.pack("<I", out_x))
            f.write(struct.pack("<I", out_y))
    size = os.path.getsize(output_path)
    avg = num_mappings / (cam_num * single_width * single_height)
    print(f"✅ 映射表完成，大小 {size} 字节 ({size/BYTES_PER_KB/BYTES_PER_KB:.{FLOAT_PRECISION}f} MB)，"
          f"平均保留比例 {avg:.{FLOAT_PRECISION}f}")


def save_column_major_map(mapping_dict, canvas_w, canvas_h, output_path, invalid_cam=65535):
    """生成列扫描顺序、仅包含 <cam_id, src_x, src_y> 的 hk8.bin。"""
    print(f"\n保存列扫描映射: {output_path}")
    table = np.zeros((canvas_w * canvas_h, 3), dtype=np.uint16)
    table[:, 0] = invalid_cam

    for (cam_id, src_x, src_y), (out_x, out_y) in mapping_dict.items():
        idx = out_x * canvas_h + out_y
        table[idx, 0] = np.uint16(cam_id)
        table[idx, 1] = np.uint16(src_x)
        table[idx, 2] = np.uint16(src_y)

    table.tofile(output_path)
    size = os.path.getsize(output_path)
    print(f"✅ 列扫描映射完成，大小 {size} 字节")


# ------------------------- 主流程 ------------------------- #
def main():
    raw_images = []
    for path in IMG_FILES:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到图像: {path}")
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"无法读取图像: {path}")
        raw_images.append(img)

    raw_h, raw_w = raw_images[0].shape[:2]
    print(f"图像尺寸: {raw_w}x{raw_h}")
    inv_map_x, inv_map_y = build_inverse_cylindrical_maps(raw_w, raw_h, FOCAL_LENGTH)
    forward_map_x, forward_map_y = build_forward_cylindrical_maps(raw_w, raw_h, FOCAL_LENGTH)

    cyl_images = []
    
    for img in raw_images:
        cyl = cylindrical_project_image(img, inv_map_x, inv_map_y)
        cyl_images.append(cyl)
    print("柱面投影完成")

    features = [preprocess_image(cyl) for cyl in cyl_images]
    transforms = compute_transforms(features)
    shift_mat, canvas_w, canvas_h = compute_canvas_bounds(cyl_images, transforms)
    warped_images, warped_masks, warp_mats = warp_all_images(cyl_images, transforms, shift_mat, canvas_w, canvas_h)
    seam_info_list, panorama = build_seam_info(warped_images, warped_masks)

    crop_start, crop_end = determine_vertical_crop(canvas_h, TARGET_CANVAS_HEIGHT)
    if crop_start > 0 or crop_end < canvas_h:
        panorama_to_save = panorama[crop_start:crop_end]
        print(f"对全景进行垂直裁剪: {crop_start} -> {crop_end} (目标高度 {crop_end - crop_start})")
   
    seam_info_dict = {info["cam_id2"]: info for info in seam_info_list}
    mapping_dict = generate_mapping_dict(forward_map_x, forward_map_y, warp_mats, seam_info_dict, canvas_w, canvas_h)
    mapping_dict, canvas_h_cropped = crop_mapping_vertically(mapping_dict, crop_start, crop_end, canvas_h)
    
    #example :hk8_path = os.path.join(OUTPUT_DIR, "hk8.bin")
    #用于你想要生成的文件名字
    hknumber_path = os.path.join(OUTPUT_DIR, "hknumber.bin")
    save_column_major_map(mapping_dict, canvas_w, canvas_h_cropped, hknumber_path)
    print("\n🎉 柱面拼接与映射全部完成！")


if __name__ == "__main__":
    main()


