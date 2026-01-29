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
import re

# 找到 build 目录下所有 csv 文件
csv_files = []
if os.path.exists("timingwatcher.txt"):
    print("发现 timingwatcher.txt，正在读取文件列表...")
    with open("timingwatcher.txt", "r") as f:
        # 读取非空行并去重
        files = [line.strip() for line in f.readlines() if line.strip()]
        # 过滤掉不存在的文件
        csv_files = [f for f in list(set(files)) if os.path.exists(f)]
else:
    print("未找到 timingwatcher.txt，搜索当前目录下最新的 CSV...")
    all_csv = glob.glob("*.csv")
    if all_csv:
        csv_files = [max(all_csv, key=os.path.getmtime)]

if not csv_files:
    raise FileNotFoundError("没有找到任何有效的 CSV 文件")

print(f"将处理以下文件: {csv_files}")

for csv_file in csv_files:
    print(f"正在处理：{csv_file}")
    df = pd.read_csv(csv_file)

    params = ['Pkt->Dec(ms)', 'Dec->Stitch(ms)', 'Stitch->Show(ms)', 'Total(ms)']

    # 数据预处理：提取 Camera ID
    if "Camera" in df.columns:
        df["CameraID"] = df["Camera"].str.extract(r'(\d+)').astype(int)
    else:
        print(f"警告: {csv_file} 中缺少 'Camera' 列，跳过")
        continue

    fig, axes = plt.subplots(len(params), 1, figsize=(12, 16), sharex=True)
    fig.suptitle(f"Timing Analysis: {csv_file}", fontsize=16)

    for i, (ax, target_param) in enumerate(zip(axes, params)):

        if target_param == 'Stitch->Show(ms)':
            # 只绘制一条线，例如 CameraID=0（所有 camera 数据一样）
            cam_df = df[df["CameraID"] == df["CameraID"].min()]
            ax.plot(
                cam_df["FrameCount"].to_numpy(), # [修改标记 1] 将 Series 转换为 numpy 数组
                cam_df[target_param].to_numpy(), # [修改标记 1] 将 Series 转换为 numpy 数组
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
                    cam_df["FrameCount"].to_numpy(), # [修改标记 2] 将 Series 转换为 numpy 数组
                    cam_df[target_param].to_numpy(), # [修改标记 2] 将 Series 转换为 numpy 数组
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
