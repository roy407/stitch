import pandas as pd
import matplotlib.pyplot as plt

# === 1️⃣ 读取 CSV 文件 ===
df = pd.read_csv("../build/2025-11-09_19-50-08.csv")

# === 2️⃣ 选择要绘制的参数 ===
# 可选值包括: 'Pkt->Dec(ms)', 'Dec->Stitch(ms)', 'Stitch->Show(ms)', 'Total(ms)'
# 注意 ！ 由于Stitch->Show(ms)的时候，就只有一张图像了，因此不同cam的结果是一样的
target_param = 'Total(ms)'  # ← 修改这里选择不同参数

# === 3️⃣ 数据预处理 ===
# 去掉 cam_ 前缀并提取编号
df["CameraID"] = df["Camera"].str.extract(r'(\d+)').astype(int)

# === 4️⃣ 绘制 ===
plt.figure(figsize=(10, 6))
for cam_id in sorted(df["CameraID"].unique()):
    cam_df = df[df["CameraID"] == cam_id]
    plt.plot(
        cam_df["FrameCount"], 
        cam_df[target_param], 
        marker='o', 
        linestyle='-', 
        linewidth=1.5,
        label=f"Camera {cam_id}"
    )

# === 5️⃣ 图形设置 ===
plt.title(f"{target_param} over Frames", fontsize=14)
plt.xlabel("Frame Count", fontsize=12)
plt.ylabel("Time (ms)", fontsize=12)
plt.legend(title="Camera")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()
