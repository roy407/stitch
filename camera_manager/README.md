# Camera Manager 模块文档

## 概述

`camera_manager` 是一个高性能的多相机视频处理框架，支持多路视频流的采集、解码、拼接和输出。该模块采用生产者-消费者模式，通过管道（Pipeline）管理多个相机的处理流程，支持 RTSP、MP4 文件和 USB 摄像头等多种输入源，并提供了基于 CUDA 的硬件加速解码和图像拼接功能。

## 核心特性

- **多输入源支持**：RTSP 流、MP4 文件、USB 摄像头
- **硬件加速**：基于 CUDA 的硬件解码和图像处理
- **多相机拼接**：支持多路相机画面的实时拼接
- **多线程架构**：基于 TaskManager 的多线程任务管理
- **灵活配置**：通过 JSON 配置文件管理相机和管道参数
- **回调机制**：支持通过 CallbackConsumer 进行帧数据回调

## 目录结构

```
camera_manager/
├── include/          # 头文件目录
│   ├── camera_manager.h      # 主管理器类
│   ├── Pipeline.h            # 管道类
│   ├── Producer.h            # 生产者基类
│   ├── PacketProducer.h      # 数据包生产者基类
│   ├── RTSPPacketProducer.h  # RTSP 流生产者
│   ├── MP4PacketProducer.h   # MP4 文件生产者
│   ├── USBPacketProducer.h   # USB 摄像头生产者
│   ├── Consumer.h            # 消费者基类
│   ├── DecoderConsumer.h      # 解码器消费者
│   ├── JetsonDecoderConsumer.h # Jetson 专用解码器
│   ├── StitchConsumer.h      # 拼接消费者
│   ├── SingleViewConsumer.h  # 单视图缩放消费者
│   ├── RtspConsumer.h        # RTSP 推流消费者
│   ├── LogConsumer.h         # 日志统计消费者
│   ├── CallbackConsumer.h    # 回调消费者
│   ├── Channel.h            # 通道类（PacketChannel, FrameChannel）
│   ├── TaskManager.h         # 任务管理器基类
│   ├── IStitch.h            # 拼接接口
│   ├── StitchImpl.h         # 拼接实现模板类
│   └── cuda_handle_init.h   # CUDA 设备初始化
└── src/              # 源文件目录
    ├── camera_manager.cpp
    ├── Pipeline.cpp
    ├── TaskManager.cpp
    ├── Producer.cpp
    ├── Consumer.cpp
    ├── PacketProducer.cpp
    ├── RTSPPacketProducer.cpp
    ├── MP4PacketProducer.cpp
    ├── USBPacketProducer.cpp
    ├── DecoderConsumer.cpp
    ├── JetsonDecoderConsumer.cpp
    ├── StitchConsumer.cpp
    ├── SingleViewConsumer.cpp
    ├── RtspConsumer.cpp
    ├── LogConsumer.cpp
    ├── CallbackConsumer.cpp
    ├── Channel.cpp
    ├── EncoderConsumer.cpp
    └── cuda_handle_init.cpp
```

## 架构设计

### 1. 核心类层次结构

```
TaskManager (基类)
├── Producer
│   └── PacketProducer
│       ├── RTSPPacketProducer
│       ├── MP4PacketProducer
│       └── USBPacketProducer
└── Consumer
    ├── DecoderConsumer
    ├── JetsonDecoderConsumer
    ├── StitchConsumer
    ├── SingleViewConsumer
    ├── RtspConsumer
    ├── LogConsumer
    └── CallbackConsumer
```

### 2. 数据流架构

```
[输入源] → [PacketProducer] → [PacketChannel] → [DecoderConsumer] → [FrameChannel]
                                                      ↓
                                              [SingleViewConsumer] → [FrameChannel] (单视图)
                                                      ↓
                                              [StitchConsumer] → [FrameChannel] (拼接图)
                                                      ↓
                                              [CallbackConsumer] (回调处理)
                                                      ↓
                                              [RtspConsumer] (RTSP 推流)
```

### 3. 管道（Pipeline）结构

每个 `Pipeline` 管理一个完整的处理流程：

1. **多个 PacketProducer**：每个相机对应一个生产者
2. **多个 DecoderConsumer**：每个相机对应一个解码器
3. **可选的 SingleViewConsumer**：单视图缩放（如果启用）
4. **一个 StitchConsumer**：多相机拼接
5. **可选的 RtspConsumer**：RTSP 推流输出
6. **可选的 CallbackConsumer**：帧数据回调

## 核心组件详解

### 1. camera_manager（主管理器）

**职责**：
- 管理所有 Pipeline 实例
- 提供单例访问接口
- 统一启动/停止所有管道
- 提供流访问接口

**关键接口**：
```cpp
static camera_manager* GetInstance();  // 获取单例
void start();                           // 启动所有管道
void stop();                            // 停止所有管道
void initPipeline();                    // 初始化管道
FrameChannel* getStitchCameraStream(int pipeline_id) const;  // 获取拼接流
FrameChannel* getSingleCameraSubStream(int cam_id) const;     // 获取单相机流
size_t getResizeCameraStreamCount() const;                    // 获取缩放流数量
```

**使用示例**：
```cpp
auto* mgr = camera_manager::GetInstance();
mgr->start();
FrameChannel* stitch_stream = mgr->getStitchCameraStream(0);
```

### 2. Pipeline（管道）

**职责**：
- 根据配置创建和管理所有 Producer 和 Consumer
- 连接各个组件之间的 Channel
- 管理整个处理流程的生命周期

**关键接口**：
```cpp
Pipeline(int pipeline_id);                    // 从 pipeline_id 构造
Pipeline(const PipelineConfig& p);             // 从配置构造
static void setLogConsumer(LogConsumer* log);  // 设置日志消费者
void start();                                  // 启动管道
void stop();                                   // 停止管道
FrameChannel* getStitchCameraStream() const;   // 获取拼接流
FrameChannel* getResizeCameraStream(int cam_id) const;  // 获取缩放流
size_t getResizeCameraStreamCount() const;      // 获取缩放流数量
```

**初始化流程**：
1. 为每个相机创建 `PacketProducer`（根据配置类型：RTSP/MP4/USB）
2. 如果启用 RTSP 推流，创建 `RtspConsumer` 并连接到 Producer
3. 为每个相机创建 `DecoderConsumer`（普通解码器或 Jetson 解码器）
4. 如果启用单视图，创建 `SingleViewConsumer`
5. 创建 `StitchConsumer` 进行多相机拼接
6. 所有组件通过 Channel 连接

### 3. TaskManager（任务管理器）

**职责**：
- 提供线程管理功能
- 统一的任务启动/停止接口
- 线程生命周期管理

**关键接口**：
```cpp
TaskManager();           // 构造函数
virtual ~TaskManager();  // 析构函数
virtual void start();    // 启动线程
virtual void stop();     // 停止线程
virtual void run() = 0;  // 纯虚函数，子类实现具体逻辑
```

**实现机制**：
- 使用 `std::thread` 管理线程
- `start()` 创建线程并调用 `run()`
- `stop()` 设置 `running=false` 并等待线程结束
- 每个线程有唯一的名称标识

### 4. PacketProducer（数据包生产者）

**职责**：
- 从输入源读取视频数据包（AVPacket）
- 将数据包发送到 `PacketChannel`
- 支持多种输入源类型

**子类实现**：

#### RTSPPacketProducer
- 从 RTSP 流读取数据
- 支持自动重连机制
- 配置选项：`buffer_size`, `rtsp_transport`, `stimeout`
- 支持 TCP 传输模式

#### MP4PacketProducer
- 从 MP4 文件读取数据
- 支持文件循环播放

#### USBPacketProducer
- 从 USB 摄像头（V4L2）读取数据
- 使用 FFmpeg 的 `v4l2` 输入格式

**数据流**：
```
输入源 → AVPacket → PacketChannel (2个)
                      ├── m_channel2rtsp (RTSP 推流)
                      └── m_channel2decoder (解码)
```

**关键接口**：
```cpp
int getWidth() const;                    // 获取宽度
int getHeight() const;                   // 获取高度
AVRational getTimeBase() const;          // 获取时间基准
AVCodecParameters* getAVCodecParameters() const;  // 获取编解码参数
PacketChannel* getChannel2Rtsp() const;  // 获取 RTSP 通道
PacketChannel* getChannel2Decoder() const;  // 获取解码通道
```

### 5. DecoderConsumer（解码器消费者）

**职责**：
- 从 `PacketChannel` 接收 AVPacket
- 使用硬件解码器（CUDA）解码为 AVFrame
- 将解码后的帧发送到 `FrameChannel`

**支持的解码器**：
- `h264_cuvid`：NVIDIA CUDA 硬件 H.264 解码器
- `hevc_cuvid`：NVIDIA CUDA 硬件 HEVC 解码器
- 其他 FFmpeg 支持的硬件解码器

**数据流**：
```
PacketChannel → AVPacket → avcodec_send_packet()
                          → avcodec_receive_frame()
                          → AVFrame (CUDA) → FrameChannel (2个)
                                              ├── m_channel2stitch (拼接)
                                              └── m_channel2resize (缩放)
```

**关键实现**：
- 使用 `cuda_handle_init` 获取 CUDA 设备句柄
- 解码后的帧格式为 `AV_PIX_FMT_CUDA`
- 为拼接和缩放分别创建帧引用（`av_frame_ref`）
- 记录解码时间戳

**关键接口**：
```cpp
DecoderConsumer(const std::string& codec_name);  // 构造函数
void setAVCodecParameters(AVCodecParameters* codecpar, AVRational time_base);  // 设置编解码参数
void setChannel(PacketChannel* channel);         // 设置输入通道
FrameChannel* getChannel2Resize();               // 获取缩放通道
FrameChannel* getChannel2Stitch();               // 获取拼接通道
```

### 6. StitchConsumer（拼接消费者）

**职责**：
- 从多个 `FrameChannel` 接收解码后的帧
- 调用 CUDA 拼接内核进行多相机拼接
- 将拼接结果发送到输出 `FrameChannel`
- 支持多个输出通道（显示和 RTSP 推流）

**关键特性**：
- 支持多种拼接模式：`mapping_table`（映射表）、`raw`（原始）
- 支持多种像素格式：`YUV420`、`YUV420P`
- 通过 `StitchOps` 接口调用模板化的拼接实现
- 支持 KERNEL_TEST 模式用于内核测试

**数据流**：
```
多个 FrameChannel → 收集所有相机帧
                → StitchOps::stitch() (CUDA 拼接)
                → AVFrame (拼接结果)
                → FrameChannel (2个输出)
                    ├── m_channel2show (显示)
                    └── m_channel2rtsp (RTSP 推流)
```

**关键接口**：
```cpp
StitchConsumer(StitchOps* ops, int single_width, int height, int output_width);  // 构造函数
void setChannels(std::vector<FrameChannel*> channels);  // 设置输入通道
FrameChannel* getChannel2Show();   // 获取显示通道
FrameChannel* getChannel2Rtsp();   // 获取 RTSP 通道
```

**实现细节**：
- 使用 `av_frame_ref` 创建帧引用，避免数据拷贝
- 记录拼接时间戳用于性能分析
- 支持优雅退出时的资源清理

### 7. SingleViewConsumer（单视图缩放消费者）

**职责**：
- 接收单个相机的解码帧
- 使用 CUDA 进行图像缩放（当前实现中缩放功能已注释）
- 将处理后的帧发送到输出 `FrameChannel`

**使用场景**：
- 单相机预览
- 低分辨率显示

**关键接口**：
```cpp
SingleViewConsumer(int width, int height, float scale_factor);  // 按比例缩放
SingleViewConsumer(int width, int height, AVRational rational);  // 按比例缩放
SingleViewConsumer(int width, int height, int output_width, int output_height);  // 指定输出尺寸
void setChannel(FrameChannel* channel);  // 设置输入通道
FrameChannel* getChannel2Show() const;   // 获取输出通道
```

### 8. CallbackConsumer（回调消费者）

**职责**：
- 从 `FrameChannel` 接收帧数据
- 通过回调函数将帧数据传递给外部处理
- 支持时间戳记录和性能分析

**设计特点**：
- 提供标准化的回调接口
- 封装帧数据的线程安全传输
- 内建时间记录机制
- 内部管理独立的消费者线程

**使用示例**：
```cpp
auto consumer = std::make_shared<CallbackConsumer>();
consumer->setChannel(frameChannel);
consumer->setCallback([](Frame frame) {
    // 处理接收到的帧数据
    processFrame(frame);
});
consumer->setPipelineName("pipeline_0");
consumer->setTimingWatcher(true);  // 启用时间监控
consumer->start();
```

**关键接口**：
```cpp
void setChannel(FrameChannel* channel);              // 设置输入通道
void setCallback(Callback_Handle callback);          // 设置回调函数
void setPipelineName(std::string name);             // 设置管道名称
void setTimingWatcher(bool enable);                 // 启用/禁用时间监控
```

### 9. Channel（通道）

**职责**：
- 提供线程安全的数据传输通道
- 基于队列实现的生产者-消费者模式

**类型**：

#### PacketChannel
```cpp
class PacketChannel {
    list_queue<Packet> m_data;  // 或 safe_queue<Packet>
public:
    bool recv(Packet& out);  // 阻塞接收
    void send(Packet& p);    // 发送
    void clear();            // 清空
    void stop();             // 停止
};
```

#### FrameChannel
```cpp
class FrameChannel {
    list_queue<Frame> m_data;  // 或 safe_queue<Frame>
public:
    bool recv(Frame& out);   // 阻塞接收
    void send(Frame& p);     // 发送
    void clear();            // 清空
    void stop();             // 停止
};
```

**特性**：
- 使用 `wait_and_pop` 实现阻塞接收
- 线程安全的数据传输
- 支持停止信号
- 支持清空队列

### 10. StitchImpl（拼接实现）

**职责**：
- 提供模板化的拼接实现
- 管理 CUDA 内存和内核参数
- 执行实际的拼接操作

**模板参数**：
- `Format`：像素格式（`YUV420`, `YUV420P`）
- `KernelTag`：拼接内核类型（`MappingTableKernel`, `RawKernel`, `HMatrixInvKernel` 等）

**支持的拼接内核**：
- `MappingTableKernel`：基于映射表的拼接（最常用）
- `RawKernel`：原始拼接
- `HMatrixInvKernel`：基于逆单应矩阵的拼接
- `HMatrixInvV1_1Kernel`：版本 1.1 的逆单应矩阵拼接
- `HMatrixInvV2Kernel`：版本 2 的逆单应矩阵拼接
- `CropKernel`：裁剪内核（未测试）

**实现流程**：
1. `init_impl()`：分配 CUDA 内存，初始化内核参数
2. `do_stitch_impl()`：
   - 复制输入帧数据到 GPU
   - 分配输出帧缓冲区
   - 调用 CUDA 内核执行拼接
   - 返回拼接结果

**StitchOps 接口**：
```cpp
struct StitchOps {
    using StitchFunc = AVFrame* (*)(void*, AVFrame**);
    using InitFunc   = void (*)(void*, int, int, int, int);
    
    void* obj = nullptr;
    StitchFunc stitch = nullptr;
    InitFunc init = nullptr;
};

// 创建 StitchOps
template<typename Impl>
StitchOps* make_stitch_ops(Impl* obj);

// 删除 StitchOps
template<typename Impl>
void delete_stitch_ops(StitchOps* ops);
```

### 11. LogConsumer（日志消费者）

**职责**：
- 定期打印 Producer 和 Consumer 的统计信息
- 监控帧率、时间戳等性能指标
- 打印 GPU/CPU 状态

**统计信息**：
- 帧计数
- 时间戳
- 帧率计算
- GPU 使用率（如果支持）

**关键接口**：
```cpp
void setProducer(PacketProducer* pro);   // 注册生产者
void setConsumer(StitchConsumer* con);   // 注册消费者
```

### 12. RtspConsumer（RTSP 推流消费者）

**职责**：
- 从 `PacketChannel` 接收数据包
- 将数据包推送到 RTSP 服务器
- 管理 RTSP 输出流

**关键接口**：
```cpp
RtspConsumer(const std::string& push_stream_url);  // 构造函数
void setChannel(PacketChannel* m_channel);         // 设置输入通道
void setParamters(AVCodecParameters* codecpar, AVRational time_base);  // 设置参数
```

### 13. JetsonDecoderConsumer（Jetson 专用解码器）

**职责**：
- 针对 Jetson 平台的专用解码器实现
- 优化 Jetson 硬件解码性能

**与 DecoderConsumer 的区别**：
- 使用 Jetson 特定的解码器实现
- 可能使用不同的硬件加速路径

## 配置系统

### 配置文件结构

通过 `CFG_HANDLE` 访问配置，配置包括：

**PipelineConfig**：
- `pipeline_id`：管道 ID
- `enable`：是否启用
- `default_width`, `default_height`：默认分辨率
- `cameras`：相机配置列表
- `stitch`：拼接配置
  - `stitch_mode`：拼接模式（`"mapping_table"`, `"raw"`）
  - `stitch_impl`：拼接实现配置

**CameraConfig**：
- `cam_id`：相机 ID
- `name`：相机名称
- `input_url`：输入 URL（RTSP/文件路径/USB 设备）
- `width`, `height`：分辨率
- `enable_view`：是否启用单视图
- `scale_factor`：缩放因子
- `rtsp`：是否启用 RTSP 推流
- `output_url`：RTSP 推流地址

**GlobalConfig**：
- `type`：输入类型（`"rtsp"`, `"mp4"`, `"usb"`）
- `format`：像素格式（`"YUV420"`, `"YUV420P"`）
- `decoder`：解码器名称（`"h264_cuvid"`, `"jetson"` 等）
- `encoder`：编码器名称

### 配置示例

```json
{
  "pipelines": [
    {
      "pipeline_id": 0,
      "enable": true,
      "default_width": 1920,
      "default_height": 1080,
      "cameras": [
        {
          "cam_id": 0,
          "name": "camera_0",
          "input_url": "rtsp://192.168.1.100:554/stream",
          "width": 3840,
          "height": 2160,
          "enable_view": true,
          "scale_factor": 0.5,
          "rtsp": false
        }
      ],
      "stitch": {
        "stitch_mode": "mapping_table",
        "stitch_impl": {
          "mapping_table": {
            "output_width": 9600,
            "d_mapping_table": "..."
          }
        }
      }
    }
  ],
  "global": {
    "type": "rtsp",
    "format": "YUV420",
    "decoder": "h264_cuvid"
  }
}
```

## 使用流程

### 1. 初始化

```cpp
#include "camera_manager.h"

// 获取单例（会自动初始化所有 Pipeline）
auto* mgr = camera_manager::GetInstance();
```

### 2. 启动

```cpp
mgr->start();  // 启动所有 Pipeline
```

### 3. 获取流

```cpp
// 获取拼接流
FrameChannel* stitch_stream = mgr->getStitchCameraStream(0);

// 获取单相机流
FrameChannel* single_stream = mgr->getSingleCameraSubStream(0);
```

### 4. 接收帧

```cpp
Frame frame;
while (stitch_stream->recv(frame)) {
    AVFrame* avframe = frame.m_data;
    // 处理帧数据...
    av_frame_free(&frame.m_data);
}
```

### 5. 使用回调消费者

```cpp
#include "CallbackConsumer.h"

auto callback_consumer = std::make_shared<CallbackConsumer>();
callback_consumer->setChannel(stitch_stream);
callback_consumer->setCallback([](Frame frame) {
    // 处理帧数据
    processFrame(frame.m_data);
    av_frame_free(&frame.m_data);
});
callback_consumer->setPipelineName("pipeline_0");
callback_consumer->start();
```

### 6. 停止

```cpp
mgr->stop();  // 停止所有 Pipeline
```

## 依赖库

### 系统库
- `pthread`：线程支持

### FFmpeg 库
- `avcodec`：编解码
- `avformat`：格式处理
- `avutil`：工具函数
- `avdevice`：设备支持（V4L2）
- `swscale`：图像缩放（部分功能）

### CUDA 库
- `cuda`：CUDA 运行时

### 项目内部库
- `config`：配置管理
- `operator_nvidia`：NVIDIA 操作符（拼接内核）
- `utils`：工具函数（包含 list_queue 或 safe_queue）

### 第三方库
- `opencv`：图像处理（部分功能，可选）
- `onnxruntime`：ONNX 推理（可选）

## 编译配置

### CMakeLists.txt 关键配置

```cmake
# 包含目录
target_include_directories(camera_manager PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/utils/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/rtsp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/operator/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/operator/include/nvidia
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/config/include
    ${FFMPEG_INCLUDE_DIRS}
    ...
)

# 链接库
target_link_libraries(camera_manager PRIVATE
    avcodec avformat avutil avdevice swscale
    pthread
    config
    cuda
    operator_nvidia
    utils
    ...
)
```

## 性能优化

### 1. 硬件加速
- 使用 CUDA 硬件解码器（`h264_cuvid`, `hevc_cuvid`）
- 所有图像处理在 GPU 上完成
- 避免 CPU-GPU 数据传输

### 2. 多线程架构
- 每个 Producer/Consumer 运行在独立线程
- 通过 Channel 实现线程间通信
- 避免锁竞争，使用无锁队列

### 3. 内存管理
- 使用 FFmpeg 的帧池（`hw_frames_ctx`）
- 复用 GPU 内存缓冲区
- 及时释放不再使用的帧
- 使用 `av_frame_ref` 避免不必要的拷贝

### 4. 帧引用机制
- 在 StitchConsumer 中使用 `av_frame_ref` 创建帧引用
- 多个输出通道共享同一帧数据
- 减少内存拷贝开销

## 常见问题

### 1. RTSP 连接失败
- 检查网络连接
- 确认 RTSP URL 正确
- 检查防火墙设置
- 代码中已实现自动重连机制

### 2. 解码失败
- 确认使用硬件解码器（`h264_cuvid`）
- 检查 CUDA 设备是否可用
- 确认输入流格式支持
- 检查 CUDA 设备句柄初始化

### 3. 拼接失败
- 检查拼接配置（映射表、输出尺寸）
- 确认所有相机帧都已接收
- 检查 CUDA 内存是否足够
- 确认拼接内核参数正确

### 4. 帧丢失
- 检查 Channel 缓冲区大小
- 确认 Consumer 处理速度足够快
- 监控帧计数统计信息
- 检查线程是否正常退出

### 5. 内存泄漏
- 确保所有 `av_frame_alloc()` 都有对应的 `av_frame_free()`
- 检查 `av_frame_ref()` 后的引用计数
- 使用 `av_frame_unref()` 减少引用计数
- 在退出时清理所有资源

### 6. 线程退出问题
- 确保 `stop()` 被正确调用
- 检查 `running` 标志是否正确设置
- 确认 Channel 的 `stop()` 被调用
- 等待所有线程正常退出

## 扩展开发

### 添加新的输入源类型

1. 继承 `PacketProducer`
2. 实现 `run()` 方法读取数据
3. 在 `Pipeline.cpp` 中添加创建逻辑：

```cpp
if(type == "new_type") {
    pro = new NewTypePacketProducer(cam);
}
```

### 添加新的拼接内核

1. 在 `IStitch.h` 中定义新的 `KernelTag` 结构体：

```cpp
struct NewKernel {
    // 内核参数
    bool initgetKernelGpuMemory(int num);
    void freeKernelGpuMemory();
};
```

2. 实现 `initgetKernelGpuMemory()` 和 `freeKernelGpuMemory()`
3. 在 `Pipeline.cpp` 的 `getStitchConsumer()` 中添加支持

### 添加新的消费者类型

1. 继承 `Consumer`
2. 实现 `run()` 方法处理帧
3. 在 `Pipeline.cpp` 中添加创建和连接逻辑

### 使用 CallbackConsumer 进行自定义处理

```cpp
auto consumer = std::make_shared<CallbackConsumer>();
consumer->setChannel(frameChannel);
consumer->setCallback([](Frame frame) {
    // 自定义处理逻辑
    customProcess(frame);
});
consumer->start();
```

## 调试技巧

### 1. 启用日志
- 使用 `LogConsumer` 监控性能
- 检查帧率和时间戳
- 监控 GPU 使用率

### 2. 时间戳分析
- 使用 `CallbackConsumer` 的 `setTimingWatcher(true)` 启用时间监控
- 分析各阶段处理时间
- 定位性能瓶颈

### 3. KERNEL_TEST 模式
- 在编译时定义 `KERNEL_TEST`
- 用于测试拼接内核性能
- 避免实际相机输入

### 4. 资源监控
- 监控 CUDA 内存使用
- 检查线程数量
- 监控 Channel 队列长度

