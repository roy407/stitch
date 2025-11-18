import cv2
import numpy as np
import json
import os

def compute_homography(img1_path, img2_path, use_flann=True):
    """计算两张图像的H矩阵: img1 -> img2"""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if use_flann:
        FLANN_INDEX_KDTREE = 1
        flann = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=50))
        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    else:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
    else:
        return None

# ------------------ 图像路径 ------------------
image_paths = [
    r"5/cam0.png",
    r"5/cam1.png"
    # r"5/cam2.png",
    # r"5/cam3.png",
    # r"5/cam4.png",
    # r"5/cam5.png",
    # r"5/cam6.png",
    # r"5/cam7.png"
]
ref_idx = 0

# ------------------ 相邻匹配 ------------------
direct_matches = [(i, i+1) for i in range(len(image_paths)-1)]
H_dict = {}
for i, j in direct_matches:
    H = compute_homography(image_paths[i], image_paths[j])
    if H is None:
        raise ValueError(f"图像 {i} 和 {j} 匹配点不足")
    H_dict[(i, j)] = H
    print(f"H_{i}->{j} 已计算")

# ------------------ 计算到参考图像的H ------------------
H_to_ref = {ref_idx: np.eye(3)}
for i in range(ref_idx-1, -1, -1):
    H_to_ref[i] = H_to_ref[i+1] @ H_dict[(i, i+1)]
for i in range(ref_idx+1, len(image_paths)):
    H_to_ref[i] = H_to_ref[i-1] @ np.linalg.inv(H_dict[(i-1, i)])

# ------------------ 构建JSON字典 ------------------
H_inv_to_ref = {}
for i in range(len(image_paths)):
    mat = np.linalg.inv(H_to_ref[i])
    # 保留4位小数或科学计数法
    formatted = [[float(f"{v:.4g}") for v in row] for row in mat]
    H_inv_to_ref[f"cam{i}"] = formatted

# ------------------ 自定义JSON写入函数 ------------------
def dump_matrix_json(data, path):
    with open(path, "w") as f:
        f.write("{\n")
        for idx, (k, v) in enumerate(data.items()):
            f.write(f'    "{k}": [\n')
            for r_i, row in enumerate(v):
                row_str = ", ".join(f"{x:.4g}" for x in row)
                comma = "," if r_i < len(v) - 1 else ""
                f.write(f"        [{row_str}]{comma}\n")
            f.write("    ]")
            if idx < len(data) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("}\n")

# ------------------ 保存JSON ------------------
json_path = r"H_inv_to_ref2.json"
dump_matrix_json(H_inv_to_ref, json_path)

print(f"已保存到: {json_path}")
