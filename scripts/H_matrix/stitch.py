import cv2
import numpy as np
import json
import os

# ------------------ 图像路径列表 ------------------
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

ref_idx = 3  # 参考图像索引
# output_folder = r"C:\Users\34139\Desktop\Trick\pinjie"

# ------------------ 读取逆矩阵 JSON ------------------
json_path = os.path.join( "H_inv_to_ref2.json")
with open(json_path, "r") as f:
    H_inv_json = json.load(f)

# 转换为 numpy 矩阵
H_inv_dict = {key: np.array(H_inv_json[key]) for key in H_inv_json}

# ------------------ 计算每张图像四角点在参考坐标系下 ------------------
corner_positions = {}
all_corners = []

for i, path in enumerate(image_paths):
    cam_name = f"cam{i}"
    img = cv2.imread(path)
    if img is None:
        print(f"无法读取图像: {path}")
        continue

    h, w = img.shape[:2]
    # 左上、右上、右下、左下
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T

    if cam_name not in H_inv_dict:
        print(f"未在JSON中找到 {cam_name}，跳过")
        continue

    # 使用逆矩阵的逆得到参考坐标下的角点
    H_to_ref = np.linalg.inv(H_inv_dict[cam_name])
    mapped = H_to_ref @ corners
    mapped /= mapped[2, :]

    corner_positions[cam_name] = mapped[:2, :].T.tolist()
    all_corners.append(mapped[:2, :].T)

# ------------------ 计算全局画布范围 ------------------
all_corners = np.vstack(all_corners)
x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)
canvas_w = x_max - x_min
canvas_h = y_max - y_min
print(f"全局画布尺寸: {canvas_w} x {canvas_h}")

# ------------------ 创建画布 ------------------
canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

# ------------------ 拼接并建立像素映射 ------------------
for i, path in enumerate(image_paths):
    cam_name = f"cam{i}"
    if cam_name not in H_inv_dict:
        continue

    img = cv2.imread(path)
    if img is None:
        continue

    h, w = img.shape[:2]
    H_inv = H_inv_dict[cam_name]

    xv, yv = np.meshgrid(np.arange(canvas_w), np.arange(canvas_h))
    ones = np.ones_like(xv)
    pts_canvas = np.stack([xv + x_min, yv + y_min, ones], axis=-1).reshape(-1, 3).T

    pts_img = H_inv @ pts_canvas
    pts_img = pts_img.astype(np.float64)
    pts_img /= pts_img[2, :]

    u = pts_img[0, :].reshape(canvas_h, canvas_w).astype(np.float32)
    v = pts_img[1, :].reshape(canvas_h, canvas_w).astype(np.float32)

    warped = cv2.remap(img, u, v, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # 后图覆盖前图
    mask = (warped.sum(axis=2) > 0)
    canvas[mask] = warped[mask]

# ------------------ 保存拼接结果 ------------------
output_image_path = os.path.join("mosaic_full.png")
cv2.imwrite(output_image_path, canvas)
print(f"拼接完成，结果已保存到: {output_image_path}")

# ------------------ 保存四角点 JSON ------------------

output_corners_path = os.path.join("mosaic_corners.json")

with open(output_corners_path, "w") as f:
    f.write("{\n")
    for idx, (k, v) in enumerate(corner_positions.items()):
        f.write(f'    "{k}": [\n')
        for r_i, row in enumerate(v):
            row_str = ", ".join(f"{x:.4g}" for x in row)  # 保留 4 位有效数字
            comma = "," if r_i < len(v) - 1 else ""
            f.write(f"        [{row_str}]{comma}\n")
        f.write("    ]")
        if idx < len(corner_positions) - 1:
            f.write(",\n")
        else:
            f.write("\n")
    f.write("}\n")

print(f"每张图像四角点已保存到: {output_corners_path}")

