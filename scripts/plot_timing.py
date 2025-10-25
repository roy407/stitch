# -*- coding: utf-8 -*-
# 或者使用简写形式
# coding=utf-8
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool  # 或者 np.bool = np.bool_
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# 找到 build 目录下所有 csv 文件
csv_files = glob.glob("build/*.csv")
if not csv_files:
    raise FileNotFoundError("build 目录下没有找到任何 CSV 文件")
latest_csv = max(csv_files, key=os.path.getmtime)
print("正在读取：", latest_csv)
df = pd.read_csv(latest_csv)

params = ['Pkt->Dec(ms)', 'Dec->Stitch(ms)', 'Stitch->Show(ms)', 'Total(ms)']

# 数据预处理：提取 Camera ID
df["CameraID"] = df["Camera"].str.extract(r'(\d+)').astype(int)

fig, axes = plt.subplots(len(params), 1, figsize=(12, 16), sharex=True)

for i, (ax, target_param) in enumerate(zip(axes, params)):

    if target_param == 'Stitch->Show(ms)':
        # 只绘制一条线，例如 CameraID=0（所有 camera 数据一样）
        cam_df = df[df["CameraID"] == df["CameraID"].min()]
        ax.plot(
            cam_df["FrameCount"],
            cam_df[target_param],
            marker='o',
            linestyle='-',
            linewidth=1.5,
            label="Stitch->Show (same for all cameras)"
        )
        ax.legend()
    else:
        # 正常绘制所有 camera
        for cam_id in sorted(df["CameraID"].unique()):
            cam_df = df[df["CameraID"] == cam_id]
            ax.plot(
                cam_df["FrameCount"],
                cam_df[target_param],
                marker='o',
                linestyle='-',
                linewidth=1.2,
                label=f"Camera {cam_id}"
            )
        # 图例只在第一张中显示
        if i == 0:
            ax.legend(title="Camera")

    ax.set_title(f"{target_param} over Frames", fontsize=14)
    ax.set_ylabel("Time (ms)")
    ax.grid(True, linestyle='--', alpha=0.6)

axes[-1].set_xlabel("Frame Count")

plt.tight_layout()
plt.show()
