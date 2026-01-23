# Core 模块文档

## 概述

`core` 模块是项目的核心基础模块，提供了配置管理、CUDA 操作符和工具函数等基础功能。该模块为整个系统提供了配置管理、高性能 CUDA 内核操作和通用工具支持。

## 核心特性

- **配置管理**：基于 JSON 的灵活配置系统
- **CUDA 加速**：高性能的 GPU 图像处理内核
- **工具函数**：日志、队列、工具函数等通用功能
- **线程安全**：提供线程安全的数据结构

## 目录结构

```
core/
├── CMakeLists.txt          # 核心模块构建配置
├── README.md              # 本文档
├── config/                # 配置管理模块
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── config.h       # 配置管理类
│   └── src/
│       └── config.cpp     # 配置实现
├── operator/              # CUDA 操作符模块
│   ├── CMakeLists.txt
│   ├── README.md
│   ├── include/
│   │   ├── nvidia_kernel.h  # CUDA 内核头文件汇总
│   │   └── nvidia/          # CUDA 内核头文件目录
│   │       ├── stitch_with_mapping_table.cuh      # 映射表拼接
│   │       ├── stitch_with_mapping_table_yuv420p.cuh
│   │       ├── stitch_raw.cuh                     # 原始拼接
│   │       ├── stitch_raw_yuv420p.cuh
│   │       ├── stitch_with_crop.cuh               # 裁剪拼接
│   │       ├── stitch_with_h_matrix.cuh            # 单应矩阵拼接
│   │       ├── h_matrix_inv/                      # 逆单应矩阵拼接
│   │       │   ├── h_matrix_inv.h
│   │       │   ├── stitch_with_h_matrix_inv.cuh
│   │       │   ├── stitch_with_h_matrix_inv_v1.1.cuh
│   │       │   └── stitch_with_h_matrix_inv_v2.cuh
│   │       ├── resize.cuh                         # 图像缩放
│   │       ├── scale.cuh                          # 图像缩放
│   │       └── yuv_2_rgb.cuh                      # YUV 转 RGB
│   └── src/
│       └── nvidia/        # CUDA 内核实现
│           ├── stitch_with_mapping_table.cu
│           ├── stitch_raw.cu
│           ├── resize.cu
│           └── ...
└── utils/                 # 工具函数模块
    ├── CMakeLists.txt
    ├── include/
    │   ├── log.hpp        # 日志系统
    │   ├── tools.hpp      # 工具函数
    │   ├── safe_queue.hpp # 线程安全队列
    │   └── safe_list.h    # 线程安全链表（未完成）
    └── src/
        └── tools.cpp      # 工具函数实现
```

## 模块详解

### 1. config 模块（配置管理）

#### 概述

`config` 模块提供了基于 JSON 的配置管理系统，支持从配置文件加载相机配置、拼接配置、全局配置等信息。使用单例模式提供全局配置访问。

#### 核心数据结构

**CameraConfig**（相机配置）：
```cpp
struct CameraConfig {
    std::string name;           // 相机名称
    int cam_id;                 // 相机 ID
    bool enable;                // 是否启用
    std::string input_url;      // 输入 URL（RTSP/文件路径/USB 设备）
    int width, height;          // 分辨率
    std::string output_url;     // 输出 URL
    bool enable_view;           // 是否启用单视图
    double scale_factor;        // 缩放因子
    bool rtsp;                  // 是否启用 RTSP 推流
};
```

**MappingTableConfig**（映射表配置）：
```cpp
struct MappingTableConfig {
    std::string file_path;              // 映射表文件路径
    cudaTextureObject_t d_mapping_table; // CUDA 纹理对象
    int output_width;                   // 输出宽度
};
```

**StitchConfig**（拼接配置）：
```cpp
struct StitchConfig {
    std::string stitch_mode;    // 拼接模式（"mapping_table", "raw" 等）
    StitchImplConfig stitch_impl; // 拼接实现配置
    std::string output_url;      // 输出 URL
    double scale_factor;        // 缩放因子
    bool rtsp;                   // 是否启用 RTSP 推流
};
```

**PipelineConfig**（管道配置）：
```cpp
struct PipelineConfig {
    std::string name;            // 管道名称
    int pipeline_id;             // 管道 ID
    bool enable;                 // 是否启用
    bool use_substream;          // 是否使用子码流
    uint64_t default_width;      // 默认宽度
    uint64_t default_height;     // 默认高度
    std::string main_stream;      // 主码流文件
    std::string sub_stream;       // 子码流文件
    std::vector<CameraConfig> cameras; // 相机列表
    StitchConfig stitch;         // 拼接配置
    bool openTimingWatcher;      // 是否启用时间监控
};
```

**GlobalConfig**（全局配置）：
```cpp
struct GlobalConfig {
    std::string loglevel;        // 日志级别
    std::string type;            // 输入类型（"rtsp", "mp4", "usb"）
    std::string format;          // 像素格式（"YUV420", "YUV420P"）
    int record_duration;         // 录制时长
    std::string record_path;     // 录制路径
    std::string decoder;         // 解码器名称
    std::string encoder;         // 编码器名称
};
```

#### 核心接口

```cpp
class config {
public:
    // 设置配置文件名称（必须在初始化前调用）
    static void SetConfigFileName(std::string cfg_name);
    
    // 获取配置文件名称
    static std::string GetConfigFileName();
    
    // 获取单例实例
    static config& GetInstance();
    
    // 获取完整配置
    const Config GetConfig() const;
    
    // 获取全局配置
    const GlobalConfig GetGlobalConfig() const;
    
    // 获取指定管道的配置
    const PipelineConfig GetPipelineConfig(int pipeline_id) const;
    
    // 获取指定管道的相机配置列表
    const std::vector<CameraConfig> GetCamerasConfig(int pipeline_id) const;
    
    // 获取指定管道的拼接配置
    const StitchConfig GetStitchConfig(int pipeline_id) const;
};
```

#### 使用示例

```cpp
// 1. 设置配置文件名称（程序启动时）
config::SetConfigFileName("config.json");

// 2. 获取配置实例
auto& cfg = config::GetInstance();

// 3. 获取全局配置
const GlobalConfig& global = cfg.GetGlobalConfig();
std::string decoder = global.decoder;

// 4. 获取管道配置
const PipelineConfig& pipeline = cfg.GetPipelineConfig(0);

// 5. 获取相机配置
const std::vector<CameraConfig>& cameras = cfg.GetCamerasConfig(0);

// 6. 使用宏定义简化访问
#define CFG_HANDLE config::GetInstance()
auto& pipeline = CFG_HANDLE.GetPipelineConfig(0);
```

#### 配置文件格式

```json
{
  "global": {
    "loglevel": "debug",
    "type": "rtsp",
    "format": "YUV420",
    "decoder": "h264_cuvid",
    "encoder": "h264_nvenc"
  },
  "pipeline": [
    {
      "name": "pipeline_0",
      "pipeline_id": 0,
      "enable": true,
      "cameras": [
        {
          "name": "camera_0",
          "cam_id": 0,
          "input_url": "rtsp://192.168.1.100:554/stream",
          "width": 3840,
          "height": 2160,
          "enable_view": true,
          "scale_factor": 0.5
        }
      ],
      "stitch": {
        "stitch_mode": "mapping_table",
        "stitch_impl": {
          "mapping_table": {
            "file_path": "mapping_table.bin",
            "output_width": 9600
          }
        }
      }
    }
  ]
}
```

#### 映射表加载

配置系统支持从文件加载 CUDA 纹理映射表：

```cpp
bool loadMappingTable(cudaTextureObject_t& tex,
                     const std::string filename,
                     uint64_t width,
                     uint64_t height);
```

映射表文件格式为二进制文件，包含 `MapEntry` 结构数组：
```cpp
struct MapEntry {
    uint16_t cam_id;  // 相机 ID
    uint16_t map_x;   // 映射 X 坐标
    uint16_t map_y;   // 映射 Y 坐标
    uint16_t pad;     // 对齐填充
};
```

### 2. operator 模块（CUDA 操作符）

#### 概述

`operator` 模块提供了基于 CUDA 的高性能图像处理内核，包括多种拼接算法、图像缩放、格式转换等功能。所有操作都在 GPU 上执行，提供极高的处理性能。

#### 支持的拼接内核

**1. 映射表拼接（Mapping Table Stitch）**

最常用的拼接方式，使用预计算的映射表进行拼接。

- `stitch_with_mapping_table.cuh`：NV12 格式（YUV420SP）
- `stitch_with_mapping_table_yuv420p.cuh`：YUV420P 格式

**接口**：
```cpp
void launch_stitch_kernel_with_mapping_table(
    uint8_t** inputs_y, uint8_t** inputs_uv,      // 输入 Y 和 UV 数据指针数组
    int* input_linesize_y, int* input_linesize_uv, // 输入步长数组
    uint8_t* output_y, uint8_t* output_uv,        // 输出 Y 和 UV 数据指针
    int output_linesize_y, int output_linesize_uv, // 输出步长
    int cam_num,                                   // 相机数量
    int single_width,                              // 单个相机宽度
    int width, int height,                         // 输出尺寸
    const cudaTextureObject_t mapping_table,       // CUDA 纹理映射表
    cudaStream_t stream1, cudaStream_t stream2);  // CUDA 流（Y 和 UV 并行处理）
```

**特点**：
- 使用 CUDA 纹理内存加速查找
- Y 和 UV 平面并行处理
- 支持多相机拼接

**2. 原始拼接（Raw Stitch）**

简单的原始拼接方式。

- `stitch_raw.cuh`：NV12 格式
- `stitch_raw_yuv420p.cuh`：YUV420P 格式

**3. 裁剪拼接（Crop Stitch）**

支持裁剪区域的拼接。

- `stitch_with_crop.cuh`

**4. 单应矩阵拼接（Homography Matrix Stitch）**

基于单应矩阵的拼接。

- `stitch_with_h_matrix.cuh`

**5. 逆单应矩阵拼接（Inverse Homography Matrix Stitch）**

基于逆单应矩阵的拼接，支持多个版本。

- `stitch_with_h_matrix_inv.cuh`：基础版本
- `stitch_with_h_matrix_inv_v1.1.cuh`：版本 1.1
- `stitch_with_h_matrix_inv_v2.cuh`：版本 2

#### 图像处理内核

**1. 图像缩放（Resize）**

```cpp
void ReSize(
    const uint8_t* pInYData, const uint8_t* pInUVData,  // 输入 Y 和 UV 数据
    int pInWidth, int pInHeight,                        // 输入尺寸
    int pInYStride, int pInUVStride,                    // 输入步长
    uint8_t* pOutYData, uint8_t* pOutUVData,            // 输出 Y 和 UV 数据
    int pOutWidth, int pOutHeight,                       // 输出尺寸
    int pOutYStride, int pOutUVStride,                  // 输出步长
    cudaStream_t stream);                               // CUDA 流
```

**2. 图像缩放（Scale）**

另一种缩放实现。

**3. YUV 转 RGB**

```cpp
// 在 yuv_2_rgb.cuh 中定义
```

#### 使用示例

```cpp
#include "nvidia_kernel.h"

// 1. 映射表拼接
launch_stitch_kernel_with_mapping_table(
    inputs_y, inputs_uv,
    input_linesize_y, input_linesize_uv,
    output_y, output_uv,
    output_linesize_y, output_linesize_uv,
    cam_num, single_width, width, height,
    mapping_table,
    stream1, stream2
);

// 2. 图像缩放
ReSize(
    input_y, input_uv,
    in_width, in_height,
    in_y_stride, in_uv_stride,
    output_y, output_uv,
    out_width, out_height,
    out_y_stride, out_uv_stride,
    stream
);
```

#### 编译配置

```cmake
enable_language(CUDA)

add_library(operator_nvidia STATIC
  src/nvidia/scale.cu
  src/nvidia/stitch_raw.cu
  src/nvidia/stitch_with_mapping_table.cu
  # ... 其他 CUDA 源文件
)

target_link_libraries(operator_nvidia
    avcodec avformat avutil swscale
    cuda nppicc nppc nppidei config
)
```

**依赖库**：
- `cuda`：CUDA 运行时
- `nppicc`, `nppc`, `nppidei`：NVIDIA 性能基元库（NPP）
- FFmpeg 库：用于帧格式处理

### 3. utils 模块（工具函数）

#### 概述

`utils` 模块提供了通用的工具函数，包括日志系统、线程安全队列、时间工具、帧处理工具等。

#### 日志系统（log.hpp）

基于 `spdlog` 的高性能日志系统。

**日志级别**：
- `LOG_DEBUG`：调试信息
- `LOG_INFO`：一般信息
- `LOG_WARN`：警告信息
- `LOG_ERROR`：错误信息

**使用示例**：
```cpp
#include "log.hpp"

LOG_DEBUG("Debug message: {}", value);
LOG_INFO("Info message: {}", value);
LOG_WARN("Warning message: {}", value);
LOG_ERROR("Error message: {}", value);
```

**日志格式**：
```
[2024-01-19 17:08:32.510][console][info][Pipeline.cpp:18][pid:153257] pipeline id : 0
```

**配置**：
- 日志级别从配置文件读取（`global.loglevel`）
- 支持：`debug`, `info`, `warn`, `error`, `critical`

**调试宏**：
```cpp
// CUDA 错误检查
CHECK_CUDA(cudaMalloc(&ptr, size));

// 空指针检查
CHECK_NULL(ptr);
CHECK_NULL_RETURN(ptr);
CHECK_NULL_RETURN_NULL(ptr);

// FFmpeg 错误检查
CHECK_FFMPEG_RETURN(av_frame_get_buffer(frame, 32));
CHECK_FFMPEG_RETURN_FUNC(ret, av_frame_get_buffer);
```

#### 线程安全队列（safe_queue.hpp）

模板化的线程安全队列，支持阻塞和非阻塞操作。

**接口**：
```cpp
template<typename T>
class safe_queue {
public:
    void push(const T& value);           // 推送元素
    bool try_pop(T& result);              // 非阻塞弹出
    bool wait_and_pop(T& result);        // 阻塞弹出
    bool wait_and_front(T& result);       // 阻塞查看队首
    bool empty() const;                   // 是否为空
    int size() const;                    // 队列大小
    void pop_and_free();                 // 弹出并释放（用于 AVFrame/AVPacket）
    void clear();                        // 清空队列
    void stop();                         // 停止队列
    
    // 统计信息
    int frames{0};      // 帧计数
    int packets{0};     // 数据包计数
    int frame_lost{0};  // 丢失帧数
    int packet_lost{0}; // 丢失数据包数
};
```

**特性**：
- 线程安全：使用互斥锁和条件变量
- 自动释放：支持 `Packet` 和 `Frame` 类型的自动释放
- 溢出保护：队列满时自动丢弃最老的元素
- 停止机制：支持优雅停止

**使用示例**：
```cpp
#include "safe_queue.hpp"

safe_queue<Frame> frame_queue;

// 生产者
Frame frame;
// ... 填充 frame ...
frame_queue.push(frame);

// 消费者（阻塞）
Frame received;
while (frame_queue.wait_and_pop(received)) {
    // 处理帧数据
    processFrame(received);
    av_frame_free(&received.m_data);
}

// 消费者（非阻塞）
Frame received;
if (frame_queue.try_pop(received)) {
    // 处理帧数据
}
```

#### 工具函数（tools.hpp）

**时间工具**：
```cpp
// 获取当前纳秒时间戳
uint64_t get_now_time();

// 生成带时间戳的文件名
std::string get_current_time_filename(const std::string& suffix = ".txt");
```

**帧处理工具**：
```cpp
// 字符串转 AVPixelFormat
AVPixelFormat transfer_string_2_AVPixelFormat(std::string format);

// 保存 NV12 帧到文件
void save_frame_as_nv12(AVFrame* frame, const std::string& filename);

// 转换 CUDA 帧到 CPU 并保存为 NV12
void transfer_and_save_cuda_nv12(AVFrame* hw_frame, const std::string& filename);

// 在 CPU 内存中创建帧
AVFrame* get_frame_on_cpu_memory(std::string format, int width, int height);

// 在 GPU 内存中创建帧
AVFrame* get_frame_on_gpu_memory(std::string format, int width, int height, AVBufferRef* av_buffer);
```

**性能统计工具**：
```cpp
// 保存时间戳到文件
void save_cost_times_to_timestamped_file(const costTimes& t, std::ofstream& ofs);

// 保存性能表格为 CSV
void save_cost_table_csv(const costTimes& t, std::ofstream& ofs);

// 打印性能时间
void printCostTimes(const costTimes& c);
```

**绘图工具**：
```cpp
// 在 NV12 帧上绘制垂直线
void draw_vertical_line_nv12(AVFrame *frame, int x, const std::string label, int fst, int Y);
```

**使用示例**：
```cpp
#include "tools.hpp"

// 获取当前时间
uint64_t now = get_now_time();

// 保存帧
save_frame_as_nv12(frame, "output.nv12");

// 转换并保存 CUDA 帧
transfer_and_save_cuda_nv12(cuda_frame, "output.nv12");

// 创建 CPU 帧
AVFrame* cpu_frame = get_frame_on_cpu_memory("YUV420", 1920, 1080);

// 创建 GPU 帧
AVFrame* gpu_frame = get_frame_on_gpu_memory("YUV420", 1920, 1080, device_handle);
```

#### 线程安全链表（safe_list.h）

**状态**：未完成实现

```cpp
class safe_list {
    ListNode* list_head;
    ListNode* list_tail;
    int size;
public:
    void clear();
};
```

## 依赖关系

### 模块间依赖

```
core/
├── config (独立)
│   └── 依赖: nlohmann_json
├── operator (依赖 config)
│   └── 依赖: cuda, npp, FFmpeg, config
└── utils (依赖 config)
    └── 依赖: spdlog, FFmpeg, cuda, config
```

### 外部依赖

**config 模块**：
- `nlohmann_json`：JSON 解析库

**operator 模块**：
- `CUDA`：CUDA 运行时
- `NPP`：NVIDIA 性能基元库（nppicc, nppc, nppidei）
- `FFmpeg`：avcodec, avformat, avutil, swscale

**utils 模块**：
- `spdlog`：日志库
- `FFmpeg`：avcodec, avformat, avutil, swscale
- `CUDA`：CUDA 运行时

## 编译配置

### 顶层 CMakeLists.txt

```cmake
set(CORE_TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(config)
add_subdirectory(operator)
add_subdirectory(utils)
```

### 各模块 CMakeLists.txt

**config**：
```cmake
add_library(config STATIC src/config.cpp)
find_package(nlohmann_json REQUIRED)
target_link_libraries(config PRIVATE nlohmann_json::nlohmann_json)
```

**operator**：
```cmake
enable_language(CUDA)
add_library(operator_nvidia STATIC ...)
target_link_libraries(operator_nvidia cuda nppicc nppc nppidei config ...)
```

**utils**：
```cmake
add_library(utils STATIC src/tools.cpp)
target_link_libraries(utils PRIVATE avcodec avformat avutil swscale ...)
```

## 使用流程

### 1. 初始化配置

```cpp
#include "config.h"

// 程序启动时设置配置文件
config::SetConfigFileName("config.json");

// 获取配置实例
auto& cfg = config::GetInstance();
```

### 2. 使用日志

```cpp
#include "log.hpp"

LOG_INFO("Application started");
LOG_DEBUG("Debug value: {}", value);
```

### 3. 使用队列

```cpp
#include "safe_queue.hpp"

safe_queue<Frame> queue;
// ... 使用队列 ...
```

### 4. 使用 CUDA 内核

```cpp
#include "nvidia_kernel.h"

// 调用拼接内核
launch_stitch_kernel_with_mapping_table(...);
```

## 性能优化

### 1. CUDA 内核优化

- **并行处理**：Y 和 UV 平面使用不同的 CUDA 流并行处理
- **纹理内存**：映射表使用 CUDA 纹理内存加速查找
- **内存合并**：优化内存访问模式
- **流并行**：使用多个 CUDA 流提高并行度

### 2. 队列优化

- **无锁设计**：在可能的情况下使用无锁数据结构
- **批量处理**：批量处理多个元素
- **溢出保护**：自动丢弃最老元素，避免内存溢出

### 3. 配置优化

- **单例模式**：避免重复加载配置
- **延迟加载**：按需加载配置项
- **缓存机制**：缓存常用配置项

## 常见问题

### 1. 配置文件加载失败

**可能原因**：
- 配置文件路径错误
- JSON 格式错误
- 文件权限问题

**解决方法**：
- 检查配置文件路径
- 验证 JSON 格式
- 检查文件权限

### 2. CUDA 内核执行失败

**可能原因**：
- CUDA 设备不可用
- 内存不足
- 内核参数错误

**解决方法**：
- 检查 CUDA 设备状态
- 检查 GPU 内存
- 验证内核参数
- 使用 `CHECK_CUDA` 宏检查错误

### 3. 队列阻塞

**可能原因**：
- 生产者停止但未调用 `stop()`
- 队列为空且无新数据

**解决方法**：
- 确保在停止时调用 `stop()`
- 检查生产者是否正常运行

### 4. 日志不输出

**可能原因**：
- 日志级别设置过高
- 日志系统未初始化

**解决方法**：
- 检查配置文件中的 `loglevel`
- 确保配置文件已加载

## 扩展开发

### 1. 添加新的配置项

1. 在 `config.h` 中添加新的配置结构
2. 在 `config.cpp` 中添加加载逻辑
3. 更新 JSON 配置文件格式

### 2. 添加新的 CUDA 内核

1. 创建 `.cuh` 头文件定义接口
2. 创建 `.cu` 源文件实现内核
3. 在 `CMakeLists.txt` 中添加源文件
4. 在 `nvidia_kernel.h` 中包含头文件

### 3. 添加新的工具函数

1. 在 `tools.hpp` 中声明函数
2. 在 `tools.cpp` 中实现函数
3. 添加必要的依赖

