# 拼接系统配置文档，json 配置的详细说明

本项目使用 JSON 格式的配置文件来管理全景拼接系统的全局设置、相机参数以及拼接算法配置。配置系统采用分层结构，由主配置文件（如 cam10.json）和子流配置文件（如 group0_mainstream.json）组成。

## 1. 配置文件结构概览

配置系统主要分为两级：
1.  **主配置文件**：定义全局参数（Global）和拼接管道（Pipeline）列表。
2.  **流配置文件**：定义具体的相机参数列表（Cameras）和拼接算法参数（Stitch）。

## 2. 主配置文件 (Root Config)

主配置文件通常位于 `resource/` 目录下（例如 `resource/cam10.json`）。

### 2.1 Global (全局配置)

定义系统的全局运行参数。

| 参数字段 | 说明 | 示例 |
| :--- | :--- | :--- |
| `loglevel` | 日志级别 (`debug`, `info`, `warn`, `error`) | `"debug"` |
| `type` | 输入源类型 (`mp4`, `rtsp`, `usb`) | `"mp4"` |
| `format` | 像素格式 (`YUV420`, `YUV420P`) | `"YUV420"` |
| `record_duration` | 录制时长（秒） | `240` |
| `record_path` | 录制文件保存路径 | `"/home/eric/mp4/"` |
| `decoder` | 硬件解码器 | `"h264_cuvid"` |
| `encoder` | 硬件编码器 | `"h264_nvenc"` |

### 2.2 Pipeline (管道配置)

定义一组或多组拼接管道。

| 参数字段 | 说明 | 示例 |
| :--- | :--- | :--- |
| `name`  | 管道名称 | `"group0"` |
| `pipeline_id` | 管道唯一ID | `0` |
| `enable` | 是否启用该管道 | `true` |
| `use_sub_input` | 是否使用子码流作为输入 | `false` |
| `main_stream`  | 主码流配置文件路径 | `"resource/cam10/group0_mainstream.json"` |
| `sub_stream`  | 子码流配置文件路径 | `"resource/cam10/group0_substream.json"` |
| `stitch` | 拼接配置 | `{"stitch_mode": "mapping_table"}` |
| └ `stitch_mode` | 拼接模式选择 (`mapping_table`, `crop`, `h_maxter_inv_v2`.) | `"mapping_table"` |

## 3. 流配置文件 (Stream Config)

流配置文件定义了具体的相机输入和拼接算法细节（例如 `resource/cam10/group0_mainstream.json`）。

### 3.1 Cameras (相机列表)

定义参与拼接的相机源。

| 参数字段 | 说明 | 示例 |
| :--- | :--- | :--- |
| `name` | 相机名称 | `"cam0"` |
| `cam_id` | 相机ID | `0` |
| `enable` | 是否启用该相机 | `true` |
| `input_url` | 输入流地址 (RTSP URL 或 文件路径) | `"rtsp://..."` |
| `width` | 输入图像宽度 | `3840` |
| `height` | 输入图像高度 | `2160` |
| `output_url` | 单路输出流地址 (可选) | `"rtsp://..."` |
| `enable_view` | 是否启用预览窗口 | `false` |
| `scale_factor` | 缩放因子 | `0.1` |
| `rtsp` | 是否为RTSP流 (标志位) | `false` |

### 3.2 Stitch (详细拼接配置)

定义拼接的具体实现参数。

| 参数字段 | 说明 | 示例 |
| :--- | :--- | :--- |
| `output_url` | 拼接后输出的 RTSP 地址 | `"rtsp://127.0.0.1:8554/stitch_group0"` |
| `stitch_impl` | 拼接实现参数容器 | - |

#### 3.2.1 Stitch Implementation (`stitch_impl`)

根据 `stitch_mode` 的不同，配置相应的参数对象。

**Mapping Table 模式 (`mapping_table`)**:

| 参数字段 | 说明 | 示例 |
| :--- | :--- | :--- |
| `file_path` | 映射表二进制文件路径 (`.bin`) | `"resource/cam10/cam8.bin"` |
| `output_width` | 拼接后的输出宽度 | `20803` |

**Homography Matrix Inverse 模式 (`H_matrix_inv`)**:

包含每个相机对应的 3x3 H矩阵数组。

```json
"H_matrix_inv": {
    "cam0": [
        [1, 0, 7463],
        [0, 1, -31.64],
        [0, 0, 1]
    ],
    ...
}
```

## 4. C++ 数据结构映射

配置文件直接映射到 `core/config/include/config.h` 中的结构体：

*   `GlobalConfig` <-> `global`
*   `PipelineConfig` <-> `pipeline` 数组元素
*   `CameraConfig` <-> `cameras` 数组元素
*   `StitchConfig` <-> `stitch`
*   `StitchImplConfig` <-> `stitch_impl`

# TODO
1. global 的参数可以覆盖其他参数，在相同参数情况下，如果global出现了，那以global的参数优先级为第一
2. enable_view 参数加入，之后加入到config中



