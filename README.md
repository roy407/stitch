# 🚀 基于异构平台的视频拼接项目

本项目旨在实现一个高性能、低延迟的**多路视频实时拼接系统**。
系统利用**CUDA GPU加速**与**多线程优化**技术，能够同时从多路相机采集视频流，进行解码、拼接、显示与性能监测。
在未来，还会加入 **华为昇腾** 、 **jetson orinX** 等异构平台。

---

## 🧩 项目特点

- 🔹 **多摄像头实时拼接**：支持多路 RTSP 或本地视频输入。
- 🔹 **GPU 加速**：基于 CUDA 实现高效视频处理与图像拼接。
- 🔹 **多线程架构**：采用生产者-消费者模型，保证数据流畅。
- 🔹 **性能监测模块（Timing Watcher）**：
  自动记录每个处理阶段的耗时（如接收、解码、拼接、显示），并输出为 CSV 文件，方便后续性能分析与可视化。
- 🔹 **Qt 界面展示**：提供实时拼接结果显示与调试界面。
- 🔹 **模块化设计**：核心逻辑与界面层完全解耦，易于扩展与维护。

---

## 🧱 前置条件

| 环境 | 最低要求 |
|------|-----------|
| NVIDIA 驱动 | ≥ 535 |
| CUDA | ≥ 11.8 |
| FFmpeg | 需要手动编译，要支持硬件编解码 |
| openGL | 任意版本 |
| Qt | ≥ 5.0 |
| spdlog | 任意版本 |

---

## ⚙️ 编译及运行步骤

```bash
# 配置环境
bash set_env.sh

# 编译并运行程序
bash start_camera.sh -c （相机配置）
```

---

## 📁 目录结构

```
stitch/
├─start_camera.sh           # 程序启动入口
├─main.cpp
├─camera_manager            # 摄像头与线程管理
├─components                # 显示界面模块
│  └─qt
├─core                      # 项目核心配置
│  ├─config                 # 读取json文件
│  ├─operator               # 算子库
│  └─utils                  # 可用工具
├─docs                      # 项目文档
├─resource                  # 内含各种相机配置文件
└─scripts                   # 脚本仓库
    ├─H_matrix              # 用于计算多张图片之间的h矩阵
    ├─mapping_table         # 用于生成多张图片生成的map表
    └─plot_timing.py        # 用于显示图像拼接过程中，各阶段耗时
```

---

## 📊 各阶段耗时曲线

1️⃣ 视频解码耗时
![Decoding Time](docs/images/Dec_2025_10_7.png)

2️⃣ 拼接阶段耗时
![Stitching Time](docs/images/Stitch_2025_10_7.png)

3️⃣ 显示阶段耗时
![Display Time](docs/images/Show_2025_10_7.png)

4️⃣ 全流程耗时
![Total Time](docs/images/Total_2025_10_7.png)

---

## 🖼️ 最终效果图

图中是使用hk5.json文件，配置生成的五路拼接图像的效果，目前平均延时可以做到≤300ms

![最终效果图](docs/images/Photo_2025_10_7.png)
