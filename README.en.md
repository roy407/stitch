# 🚀 Video Stitching Project Based on Heterogeneous Platform

<p align="center">
  <a href="README.md">中文</a> ｜ 
  <a href="README.en.md">English</a>
</p>

This project aims to implement a high-performance, low-latency **multi-stream video real-time stitching system**.<br>
The system utilizes **CUDA GPU acceleration** and **multi-threading optimization** technologies, capable of simultaneously capturing video streams from multiple cameras, performing decoding, stitching, display, encoding, RTSP streaming, and performance monitoring.<br>
Currently adapted for **regular home computers** and **Jetson OrinX** platforms.<br>
In the future, **Huawei Ascend** and other heterogeneous platforms will be added.

---

## 📢 News | Project Progress

2025-02-01: (Expected to complete code v2.0).<br>
2025-12-02: Completed exhibition at **2025 China International Maritime Exhibition**.<br>
2025-10-06: Completed code v1.0.<br>
2025-05-10: Project initiated, aiming to solve the problem of small target recognition on the sea surface, thus requiring multiple cameras and high resolution.

---

## 🧩 Project Features

- 🔹 **Multi-Camera Real-time Stitching**: Supports multiple RTSP, USB, or local video inputs.
- 🔹 **GPU Acceleration**: Efficient video processing and image stitching based on CUDA.
- 🔹 **Hardware Codec**: Uses cuvid and nvenc for codec acceleration (uses nvmpi on OrinX).
- 🔹 **Multi-threaded Architecture**: Adopts producer-consumer model to ensure smooth data flow.
- 🔹 **Performance Monitoring Module (Timing Watcher)**:
  Automatically records the time consumption of each processing stage (such as receiving, decoding, stitching, display) and outputs as CSV files for subsequent performance analysis and visualization.
- 🔹 **Qt Interface Display**: Provides real-time stitching result display and debugging interface.
- 🔹 **Modular Design**: Core logic and interface layer are completely decoupled, easy to extend and maintain.

---

## 🧱 Prerequisites

| Environment | Minimum Requirements |
|-------------|----------------------|
| NVIDIA Driver | ≥ 535 |
| CUDA | ≥ 11.8 |
| FFmpeg | Manual compilation required, must support hardware codec |
| OpenGL | Any version |
| Qt | ≥ 5.0 |
| spdlog | Any version |

---

## ⚙️ Build and Run Steps

```bash
# Configure environment
bash set_env.sh

# Build and run program
bash start_camera.sh -c (camera configuration)
```

For more detailed operations, please see [How to Get Started with This Project](docs/入门该项目.md)

---

## 📁 Directory Structure

```
stitch/
├─start_camera.sh           # Program startup entry
├─main.cpp
├─camera_manager            # Camera and thread management
├─components                # Component modules
│  ├─qt                     # Qt display interface
│  └─shm                    # Shared memory module (not yet open)
├─core                      # Project core configuration
│  ├─config                 # Read JSON files
│  ├─operator               # Operator library
│  └─utils                  # Available utilities
├─docs                      # Project documentation
├─resource                  # Contains various camera configuration files
└─scripts                   # Script repository
    ├─H_matrix              # Used to calculate H matrix between multiple images
    ├─mapping_table         # Used to generate mapping table for multiple images
    └─plot_timing.py        # Used to display time consumption of each stage in image stitching process
```

---

## 📚 Module Documentation Navigation

- 📷 [camera_manager | Camera and Thread Management](camera_manager/README.en.md)
- 🧠 [core | Operator Library and Core Functions](core/README.en.md)
- 🧩 [components | Functional Components (Qt / shm)](components/README.en.md)
- 🗂️ [resource | Camera and Stitching Configuration Files](resource/README.en.md)

---

## 📊 Time Consumption Curves for Each Stage

1️⃣ Video Decoding Time
![Decoding Time](docs/images/Dec_2025_10_7.png)

2️⃣ Stitching Stage Time
![Stitching Time](docs/images/Stitch_2025_10_7.png)

3️⃣ Display Stage Time
![Display Time](docs/images/Show_2025_10_7.png)

4️⃣ Total Process Time
![Total Time](docs/images/Total_2025_10_7.png)

---

## 🖼️ Final Result Image

The image shows the effect of a five-way stitched image generated using the cam5.json configuration file. Currently, the average latency can be ≤300ms

![Final Result Image](docs/images/Photo_2025_10_7.png)

