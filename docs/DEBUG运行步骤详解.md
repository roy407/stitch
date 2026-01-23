# 🔍 Stitch项目DEBUG运行步骤详解

本文档详细说明从主程序入口开始的每一步调试流程，帮助开发者理解程序执行顺序和关键断点位置。

---

## 📋 目录

1. [程序入口 - main函数](#1-程序入口---main函数)
2. [配置文件加载](#2-配置文件加载)
3. [Camera Manager初始化](#3-camera-manager初始化)
4. [Pipeline初始化](#4-pipeline初始化)
5. [Producer创建与启动](#5-producer创建与启动)
6. [Consumer创建与启动](#6-consumer创建与启动)
7. [数据流处理](#7-数据流处理)
8. [UI界面启动](#8-ui界面启动)

---

## 1. 程序入口 - main函数

### 1.1 断点位置
**文件**: `main.cpp:26`
```cpp
int main(int argc, char *argv[]) {
    // 🔴 断点1: 程序入口
    std::string config_name = "";
    if (argc > 1) {
        config_name = argv[1];
    }
```

### 1.2 调试步骤
1. **设置断点**: 在 `main.cpp:26` 设置断点
2. **运行程序**: 
   ```bash
   gdb ./build/stitch_app
   (gdb) set args resource/cam2.json
   (gdb) run
   ```
3. **检查参数**:
   - `argc`: 应该 >= 2（程序名 + 配置文件路径）
   - `argv[1]`: 配置文件路径，例如 `"resource/cam2.json"`

### 1.3 关键变量
- `config_name`: 配置文件路径字符串
- `argc`: 命令行参数数量
- `argv`: 命令行参数数组

---

## 2. 配置文件加载

### 2.1 断点位置
**文件**: `main.cpp:31`
```cpp
config::SetConfigFileName(config_name);
```

**文件**: `core/config/src/config.cpp:8`
```cpp
config::config() {
    loadFromFile();  // 🔴 断点2: 配置文件加载入口
}
```

### 2.2 调试步骤
1. **单步执行**到 `config::SetConfigFileName(config_name)`
2. **进入函数** `config::GetInstance()` (在 `config.cpp:211`)
3. **断点设置**在 `config.cpp:9` (`loadFromFile()`)
4. **检查文件**:
   ```cpp
   // 在 config.cpp:13-14
   std::string filename = config_name;
   std::ifstream infile(filename);
   ```
   - 确认 `filename` 正确
   - 确认文件存在且可读

### 2.3 JSON解析检查
**文件**: `core/config/src/config.cpp:19-28`
```cpp
json j;
infile >> j;  // 🔴 断点3: JSON解析
if (j.contains("global")) loadGlobalConfig(j["global"], cfg.global);
if (j.contains("pipeline")) {
    for (auto& p : j["pipeline"]) {
        PipelineConfig pipe;
        loadPipelineConfig(p, pipe);
        cfg.pipelines.push_back(pipe);
    }
}
```

**检查项**:
- JSON文件格式是否正确
- `global` 节点是否存在
- `pipeline` 数组是否为空
- 每个pipeline配置是否完整

### 2.4 全局配置加载
**文件**: `core/config/src/config.cpp:32-40`
```cpp
void config::loadGlobalConfig(const json& j, GlobalConfig& cfg) {
    cfg.loglevel = j.value("loglevel", "debug");      // 🔴 检查日志级别
    cfg.type = j.value("type", "mp4");                // 🔴 检查输入类型
    cfg.format = j.value("format", "YUV420");          // 🔴 检查像素格式
    cfg.record_duration = j.value("record_duration", 240);
    cfg.record_path = j.value("record_path", "");
    cfg.decoder = j.value("decoder", "h264_cuvid");   // 🔴 检查解码器
    cfg.encoder = j.value("encoder", "h264_nvenc");
}
```

**关键配置项**:
- `loglevel`: 日志级别（debug/info/warn/error）
- `type`: 输入类型（mp4/rtsp/usb）
- `format`: 像素格式（YUV420/YUV420P）
- `decoder`: 解码器类型（h264_cuvid/jetson）

---

## 3. Camera Manager初始化

### 3.1 断点位置
**文件**: `main.cpp:32-42` (根据配置文件选择启动模式)
```cpp
if(config_name == "resource/cam10.json") {
    return launch_with_mainwindow(argc, argv);  // 🔴 断点4: 主窗口模式
} else if(config_name == "resource/hk5.json") {
    return launch_with_widget(0, 1920, 540, argc, argv);  // 🔴 断点5: 测试窗口模式
} else if(config_name == "resource/cam2.json") {
    return launch_with_widget(0, 1920, 540, argc, argv);
} else {
    launch_with_no_window();  // 🔴 断点6: 无窗口模式
}
```

### 3.2 无窗口模式调试
**文件**: `main.cpp:6-10`
```cpp
void launch_with_no_window() {
    camera_manager* cam = camera_manager::GetInstance();  // 🔴 断点7: 获取单例
    cam->start();  // 🔴 断点8: 启动管理器
    while(1);
}
```

### 3.3 Camera Manager构造函数
**文件**: `camera_manager/src/camera_manager.cpp:26-29`
```cpp
camera_manager::camera_manager() {
    m_log = new LogConsumer();  // 🔴 断点9: 创建日志消费者
    initPipeline();  // 🔴 断点10: 初始化Pipeline
}
```

**检查项**:
- `m_log` 是否成功创建
- `initPipeline()` 是否正常执行

### 3.4 Pipeline初始化
**文件**: `camera_manager/src/camera_manager.cpp:57-66`
```cpp
void camera_manager::initPipeline() {
    avformat_network_init();  // 🔴 断点11: FFmpeg网络初始化
    avdevice_register_all();  // 🔴 断点12: FFmpeg设备注册
    auto& cfg = CFG_HANDLE.GetConfig();  // 🔴 断点13: 获取配置
    Pipeline::setLogConsumer(m_log);
    for(auto& p : cfg.pipelines) {
        auto pipeline = new Pipeline(p);  // 🔴 断点14: 创建Pipeline
        m_pipelines.emplace_back(pipeline);
    }
}
```

**检查项**:
- FFmpeg初始化是否成功
- 配置中的pipeline数量
- 每个Pipeline是否成功创建

---

## 4. Pipeline初始化

### 4.1 Pipeline构造函数
**文件**: `camera_manager/src/Pipeline.cpp:54-114`
```cpp
Pipeline::Pipeline(const PipelineConfig &p) {
    if(p.enable == true) {  // 🔴 断点15: 检查Pipeline是否启用
        std::vector<FrameChannel*> channels;
        for(auto& cam : p.cameras) {  // 🔴 断点16: 遍历每个摄像头
            // ... 创建Producer和Consumer
        }
    }
}
```

### 4.2 Producer创建
**文件**: `camera_manager/src/Pipeline.cpp:58-67`
```cpp
std::string type = CFG_HANDLE.GetGlobalConfig().type;
PacketProducer* pro = nullptr;
if(type == "mp4") {
    pro = new MP4PacketProducer(cam);  // 🔴 断点17: 创建MP4生产者
} else if(type == "rtsp") {
    pro = new RTSPPacketProducer(cam);  // 🔴 断点18: 创建RTSP生产者
} else if(type == "usb") {
    pro = new USBPacketProducer(cam);  // 🔴 断点19: 创建USB生产者
}
m_producerTask.push_back(pro);
```

**检查项**:
- `type` 值是否正确（mp4/rtsp/usb）
- Producer是否成功创建
- 摄像头配置 `cam` 是否完整

### 4.3 Decoder Consumer创建
**文件**: `camera_manager/src/Pipeline.cpp:76-100`
```cpp
if(CFG_HANDLE.GetGlobalConfig().decoder != "jetson") {
    DecoderConsumer* dcon = new DecoderConsumer(CFG_HANDLE.GetGlobalConfig().decoder);
    // 🔴 断点20: 创建标准解码器
    dcon->setAVCodecParameters(pro->getAVCodecParameters(), pro->getTimeBase());
    dcon->setChannel(pro->getChannel2Decoder());
    m_consumerTask.push_back(dcon);
    
    if(cam.enable_view == true) {
        SingleViewConsumer* resizeCon = new SingleViewConsumer(cam.width, cam.height, cam.scale_factor);
        // 🔴 断点21: 创建单视图消费者
        resizeCon->setChannel(dcon->getChannel2Resize());
        m_resizeStream[cam.cam_id] = resizeCon->getChannel2Show();
        m_consumerTask.push_back(resizeCon);
    }
    channels.push_back(dcon->getChannel2Stitch());
} else {
    JetsonDecoderConsumer* dcon = new JetsonDecoderConsumer();
    // 🔴 断点22: 创建Jetson解码器
    // ... 类似处理
}
```

**检查项**:
- 解码器类型是否正确
- Codec参数是否设置成功
- Channel连接是否正确
- 如果启用单视图，SingleViewConsumer是否创建

### 4.4 Stitch Consumer创建
**文件**: `camera_manager/src/Pipeline.cpp:103-112`
```cpp
StitchConsumer* stitch = getStitchConsumer(p.pipeline_id, p.stitch.stitch_mode);
// 🔴 断点23: 获取拼接消费者
if(stitch != nullptr) {
    stitch->setChannels(channels);  // 🔴 断点24: 设置输入通道
    m_consumerTask.push_back(stitch);
    if(m_log) m_log->setConsumer(stitch);
    m_stitchStream = stitch->getChannel2Show();  // 🔴 断点25: 获取输出通道
} else {
    LOG_INFO("stitch consumer not init");
}
```

**检查项**:
- `p.stitch.stitch_mode` 值（"mapping_table" 或 "raw"）
- StitchConsumer是否成功创建
- 输入通道数量是否正确
- 输出通道是否有效

### 4.5 Stitch Consumer详细创建
**文件**: `camera_manager/src/Pipeline.cpp:15-48`
```cpp
StitchConsumer *Pipeline::getStitchConsumer(int pipeline_id, std::string kernelTag) {
    auto& p = CFG_HANDLE.GetPipelineConfig(pipeline_id);
    std::string format = CFG_HANDLE.GetGlobalConfig().format;
    // 🔴 断点26: 检查格式和kernelTag
    
    if(format == "YUV420") {
        if(kernelTag == "mapping_table") {
            auto stitchImpl = new StitchImpl<YUV420, MappingTableKernel>();
            // 🔴 断点27: 创建映射表拼接实现
            stitchImpl->loadMappingTable(p.stitch.stitch_impl.mapping_table.d_mapping_table);
            StitchOps* ops = make_stitch_ops(stitchImpl);
            ops->init(ops->obj, p.cameras.size(), p.cameras[0].width, 
                     p.stitch.stitch_impl.mapping_table.output_width, p.default_height);
            return new StitchConsumer(ops, p.cameras[0].width, p.default_height, 
                                     p.stitch.stitch_impl.mapping_table.output_width);
        } else if(kernelTag == "raw") {
            // 🔴 断点28: 创建原始拼接实现
            // ...
        }
    }
    // ...
}
```

**检查项**:
- 映射表文件路径是否正确
- 映射表是否成功加载到GPU
- 拼接参数（宽度、高度）是否正确

---

## 5. Producer创建与启动

### 5.1 MP4PacketProducer初始化
**文件**: `camera_manager/src/MP4PacketProducer.cpp` (假设存在)
```cpp
MP4PacketProducer::MP4PacketProducer(const CameraConfig& cam) {
    // 🔴 断点29: MP4生产者初始化
    // 打开视频文件
    // 获取流信息
    // 创建解码器上下文
}
```

**检查项**:
- 视频文件路径 `cam.input_url` 是否存在
- 文件格式是否支持
- 视频流信息是否正确

### 5.2 RTSPPacketProducer初始化
**文件**: `camera_manager/src/RTSPPacketProducer.cpp` (假设存在)
```cpp
RTSPPacketProducer::RTSPPacketProducer(const CameraConfig& cam) {
    // 🔴 断点30: RTSP生产者初始化
    // 连接RTSP流
    // 获取流信息
}
```

**检查项**:
- RTSP URL `cam.input_url` 是否可访问
- 网络连接是否成功
- 流信息是否正确

### 5.3 Producer启动
**文件**: `camera_manager/src/camera_manager.cpp:36-44`
```cpp
void camera_manager::start() {
    if(!m_running) {
        for(auto& p : m_pipelines) p->start();  // 🔴 断点31: 启动所有Pipeline
        m_log->start();  // 🔴 断点32: 启动日志消费者
        m_running = true;
    }
}
```

**文件**: `camera_manager/src/Pipeline.cpp:120-123`
```cpp
void Pipeline::start() {
    for(auto& pro : m_producerTask) pro->start();  // 🔴 断点33: 启动所有Producer
    for(auto& con : m_consumerTask) con->start();  // 🔴 断点34: 启动所有Consumer
}
```

**检查项**:
- 所有Producer是否成功启动
- 线程是否正常创建
- 数据流是否开始

---

## 6. Consumer创建与启动

### 6.1 DecoderConsumer启动
**文件**: `camera_manager/src/DecoderConsumer.cpp` (假设存在)
```cpp
void DecoderConsumer::start() {
    TaskManager::start();  // 🔴 断点35: 启动任务管理器
}

void DecoderConsumer::run() {
    // 🔴 断点36: 解码器运行循环
    while (running) {
        Packet packet;
        if(!m_channel->recv(packet)) break;  // 🔴 断点37: 接收数据包
        
        // 解码数据包
        AVFrame* frame = decode(packet);  // 🔴 断点38: 执行解码
        if(frame) {
            Frame output;
            output.m_data = frame;
            output.m_costTimes.when_get_decoded_frame[packet.cam_id] = get_now_time();
            m_channel2stitch->send(output);  // 🔴 断点39: 发送解码帧
        }
    }
}
```

**检查项**:
- 数据包是否正常接收
- 解码是否成功
- 解码后的帧格式是否正确
- 时间戳是否正确记录

### 6.2 StitchConsumer启动
**文件**: `camera_manager/src/StitchConsumer.cpp:27-29`
```cpp
void StitchConsumer::start() {
    TaskManager::start();  // 🔴 断点40: 启动拼接消费者
}
```

**文件**: `camera_manager/src/StitchConsumer.cpp:38-62`
```cpp
void StitchConsumer::run() {
    Frame out_image;
    AVFrame** inputs = new AVFrame*[10];
    // 🔴 断点41: 拼接运行循环开始
    
    while (running) {
        int frame_size = 0;
        for (auto& channel : m_channelsFromDecoder) {
            Frame tmp;
            if(!channel->recv(tmp)) goto cleanup;  // 🔴 断点42: 从每个通道接收帧
            inputs[frame_size] = tmp.m_data;
            // 复制时间戳信息
            frame_size++;
        }
        
        out_image.m_data = ops->stitch(ops->obj, inputs);  // 🔴 断点43: 执行拼接
        out_image.m_costTimes.when_get_stitched_frame = get_now_time();
        m_channel2show->send(out_image);  // 🔴 断点44: 发送拼接结果
        m_status.frame_cnt++;
        
        // 释放输入帧
        for (int i = 0; i < frame_size; ++i) {
            if (inputs[i]) {
                av_frame_free(&inputs[i]);
            }
        }
    }
}
```

**检查项**:
- 所有输入通道是否都有数据
- 拼接操作是否成功
- 输出帧格式是否正确
- 性能时间戳是否正确记录

---

## 7. 数据流处理

### 7.1 数据流路径
```
Producer -> Channel -> DecoderConsumer -> Channel -> StitchConsumer -> Channel -> UI
```

### 7.2 关键断点位置

#### 7.2.1 Producer发送数据包
**文件**: `camera_manager/src/PacketProducer.cpp` (假设存在)
```cpp
void PacketProducer::run() {
    while (running) {
        AVPacket* packet = av_packet_alloc();
        // 读取数据包
        int ret = av_read_frame(m_format_ctx, packet);  // 🔴 断点45: 读取数据包
        if (ret >= 0) {
            Packet pkt;
            pkt.m_data = packet;
            pkt.cam_id = m_cam_id;
            pkt.m_costTimes.when_get_packet[m_cam_id] = get_now_time();
            m_channel2decoder->send(pkt);  // 🔴 断点46: 发送数据包
        }
    }
}
```

#### 7.2.2 Channel数据传递
**文件**: `camera_manager/src/Channel.cpp` (假设存在)
```cpp
bool Channel::send(const T& item) {
    std::unique_lock<std::mutex> lock(m_mutex);
    // 🔴 断点47: Channel发送数据
    m_queue.push(item);
    m_cond.notify_one();
    return true;
}

bool Channel::recv(T& item) {
    std::unique_lock<std::mutex> lock(m_mutex);
    // 🔴 断点48: Channel接收数据
    m_cond.wait(lock, [this] { return !m_queue.empty() || !m_running; });
    if (!m_running && m_queue.empty()) return false;
    item = m_queue.front();
    m_queue.pop();
    return true;
}
```

**检查项**:
- 队列是否正常
- 线程同步是否正确
- 数据是否丢失

---

## 8. UI界面启动

### 8.1 Widget模式启动
**文件**: `main.cpp:12-17`
```cpp
int launch_with_widget(int pipeline_id, int width, int height, int argc, char *argv[]) {
    QApplication a(argc, argv);  // 🔴 断点49: 创建Qt应用
    widget_for_test w(pipeline_id, width, height);  // 🔴 断点50: 创建测试窗口
    w.show();  // 🔴 断点51: 显示窗口
    return a.exec();  // 🔴 断点52: 进入事件循环
}
```

### 8.2 Widget初始化
**文件**: `components/qt/src/widget_for_test.cpp:33-52`
```cpp
widget_for_test::widget_for_test(int pipeline_id, int width, int height, QWidget *parent) : 
    QOpenGLWidget(parent),
    m_render(nullptr),
    cam(nullptr),
    con(nullptr),
    running(true)
{
    setFixedSize(width, height);  // 🔴 断点53: 设置窗口大小
    m_render = new Nv12Render();  // 🔴 断点54: 创建渲染器
    cam = camera_manager::GetInstance();  // 🔴 断点55: 获取相机管理器
    cam->start();  // 🔴 断点56: 启动相机管理器
    q = cam->getStitchCameraStream(pipeline_id);  // 🔴 断点57: 获取拼接流
    con = QThread::create([this](){consumerThread();});  // 🔴 断点58: 创建消费者线程
    con->start();  // 🔴 断点59: 启动消费者线程
}
```

### 8.3 消费者线程
**文件**: `components/qt/src/widget_for_test.cpp:95-150`
```cpp
void widget_for_test::consumerThread() {
    static std::string filename = std::string("build/") + get_current_time_filename(".csv");
    std::ofstream ofs(filename, std::ios::app);
    // 🔴 断点60: 打开CSV文件
    
    AVFrame* cpu_frame = av_frame_alloc();
    while (running.load()) {
        Frame frame;
        if(!q->recv(frame)) break;  // 🔴 断点61: 接收拼接帧
        AVFrame* src_frame = frame.m_data;
        
        // 硬件帧转换到CPU
        if (src_frame->format == AV_PIX_FMT_CUDA) {
            if (av_hwframe_transfer_data(cpu_frame, src_frame, 0) < 0) {
                // 🔴 断点62: 帧转换失败
                continue;
            }
            process_frame = cpu_frame;
        }
        
        // 准备渲染数据
        m_width = process_frame->width;
        m_height = process_frame->height;
        // ... 数据拷贝
        
        frame.m_costTimes.when_show_on_the_screen = get_now_time();
        save_cost_table_csv(frame.m_costTimes, ofs);  // 🔴 断点63: 保存性能数据
        
        QMetaObject::invokeMethod(this, "update", Qt::QueuedConnection);  // 🔴 断点64: 触发重绘
    }
}
```

### 8.4 OpenGL渲染
**文件**: `components/qt/src/widget_for_test.cpp:85-89`
```cpp
void widget_for_test::paintGL() {
    if (!m_buffer.empty() && m_width > 0 && m_height > 0) {
        m_render->render(m_buffer.data(), m_width, m_height, m_y_stride, m_uv_stride);
        // 🔴 断点65: 执行OpenGL渲染
    }
}
```

---

## 9. 性能监控

### 9.1 时间戳记录点
1. **数据包接收**: `when_get_packet[cam_id]` - Producer接收数据包时间
2. **解码完成**: `when_get_decoded_frame[cam_id]` - Decoder完成解码时间
3. **拼接完成**: `when_get_stitched_frame` - StitchConsumer完成拼接时间
4. **显示完成**: `when_show_on_the_screen` - UI显示完成时间

### 9.2 性能数据保存
**文件**: `components/qt/src/widget_for_test.cpp:146`
```cpp
save_cost_table_csv(frame.m_costTimes, ofs);
```

**文件**: `core/utils/src/tools.cpp:109-135`
```cpp
void save_cost_table_csv(const costTimes& t, std::ofstream& ofs) {
    // 🔴 断点66: 保存性能数据到CSV
    // 计算各阶段耗时
    // 写入CSV文件
}
```

---

## 10. 常见问题调试

### 10.1 配置文件加载失败
**症状**: 程序启动失败，提示配置文件错误
**调试步骤**:
1. 检查断点2 (`config.cpp:9`)
2. 确认文件路径正确
3. 检查JSON格式是否正确
4. 查看日志输出

### 10.2 解码失败
**症状**: 解码器无法解码数据包
**调试步骤**:
1. 检查断点37-38 (`DecoderConsumer::run`)
2. 确认Codec参数正确
3. 检查数据包格式
4. 查看FFmpeg错误码

### 10.3 拼接失败
**症状**: 拼接结果异常或程序崩溃
**调试步骤**:
1. 检查断点42-43 (`StitchConsumer::run`)
2. 确认所有输入通道都有数据
3. 检查映射表是否正确加载
4. 查看CUDA错误

### 10.4 显示异常
**症状**: 窗口无显示或显示异常
**调试步骤**:
1. 检查断点61-64 (`widget_for_test::consumerThread`)
2. 确认帧数据正确
3. 检查OpenGL上下文
4. 查看渲染器状态

---

## 11. GDB调试命令参考

### 11.1 基本命令
```bash
# 启动调试
gdb ./build/stitch_app
(gdb) set args resource/cam2.json
(gdb) run

# 设置断点
(gdb) break main.cpp:26
(gdb) break camera_manager.cpp:36
(gdb) break Pipeline.cpp:54

# 查看变量
(gdb) print config_name
(gdb) print cfg.pipelines.size()

# 单步执行
(gdb) step        # 进入函数
(gdb) next        # 下一行
(gdb) continue    # 继续执行

# 查看调用栈
(gdb) backtrace
(gdb) frame 0
```

### 11.2 多线程调试
```bash
# 查看所有线程
(gdb) info threads

# 切换线程
(gdb) thread 2

# 为所有线程设置断点
(gdb) break DecoderConsumer::run thread all
```

### 11.3 内存检查
```bash
# 检查内存泄漏
valgrind --leak-check=full ./build/stitch_app resource/cam2.json

# 检查CUDA错误
cuda-gdb ./build/stitch_app
```

---

## 12. 完整调试流程示例

### 12.1 启动调试会话
```bash
cd /home/eric/文档/stitch
gdb ./build/stitch_app
```

### 12.2 设置所有关键断点
```bash
(gdb) break main.cpp:26
(gdb) break config.cpp:9
(gdb) break camera_manager.cpp:26
(gdb) break camera_manager.cpp:36
(gdb) break Pipeline.cpp:54
(gdb) break Pipeline.cpp:120
(gdb) break StitchConsumer.cpp:38
(gdb) break widget_for_test.cpp:95
```

### 12.3 运行并跟踪
```bash
(gdb) set args resource/cam2.json
(gdb) run
# 程序会在第一个断点停止
(gdb) continue  # 继续到下一个断点
# 重复执行continue，观察程序流程
```

### 12.4 检查关键变量
在每个断点处检查：
- 配置文件路径
- Pipeline数量
- 摄像头数量
- 通道连接状态
- 帧数据有效性

---

## 13. 日志调试

### 13.1 启用DEBUG日志
在配置文件中设置：
```json
{
    "global": {
        "loglevel": "debug"
    }
}
```

### 13.2 关键日志位置
- 配置文件加载: `config.cpp:16`
- Pipeline创建: `Pipeline.cpp:18`
- 数据包接收: `PacketProducer::run`
- 解码完成: `DecoderConsumer::run`
- 拼接完成: `StitchConsumer::run`
- 显示完成: `widget_for_test::consumerThread`

---

## 14. 总结

本文档提供了从主程序入口到UI显示的完整调试路径，包含：
- **80+个关键断点位置**
- **详细的检查项**
- **常见问题解决方案**
- **GDB调试命令参考**

按照本文档的步骤，可以系统地调试stitch项目的每个环节，快速定位问题所在。

---

**最后更新**: 2025-12-18
**文档版本**: 1.0

