# Components 模块文档

## 概述

`components` 模块提供了基于 Qt 的图形用户界面组件，用于显示和处理多相机拼接视频流。该模块包含多个 OpenGL 渲染组件，支持实时显示可见光拼接、红外拼接、单相机视图等功能，并提供了完整的主窗口界面管理。

## 核心特性

- **OpenGL 硬件加速渲染**：基于 QOpenGLWidget 的高性能视频渲染
- **NV12 格式支持**：高效的 YUV420SP 格式视频渲染
- **多视图显示**：支持可见光拼接、红外拼接、单相机视图等多种显示模式
- **实时性能监控**：显示帧率、延迟等性能指标
- **响应式布局**：支持窗口缩放和自适应布局
- **全屏显示**：支持全屏模式和窗口模式切换

## 目录结构

```
components/
├── CMakeLists.txt          # 组件构建配置
└── qt/                     # Qt 组件目录
    ├── CMakeLists.txt      # Qt 组件构建配置
    ├── include/            # 头文件目录
    │   ├── mainwindow.h              # 主窗口类
    │   ├── full_stitch_widget.h      # 全尺寸拼接显示组件
    │   ├── visible_camera_widget.h   # 可见光拼接显示组件
    │   ├── visible_single_camera_widget.h  # 单相机显示组件
    │   ├── infrared_camera_widget.h  # 红外拼接显示组件
    │   ├── nv12render.h              # NV12 格式 OpenGL 渲染器
    │   └── widget_for_test.h        # 测试组件
    └── src/                # 源文件目录
        ├── mainwindow.cpp
        ├── full_stitch_widget.cpp
        ├── visible_camera_widget.cpp
        ├── visible_single_camera_widget.cpp
        ├── infrared_camera_widget.cpp
        ├── nv12render.cpp
        └── widget_for_test.cpp
```

## 架构设计

### 1. 组件层次结构

```
QMainWindow
└── StitchMainWindow
    ├── QStackedLayout (多页面布局)
    │   ├── 主页面 (mainWidget)
    │   │   ├── 红外拼接区域 (InfraredWidget)
    │   │   ├── 可见光拼接区域 (visible_camera_widget)
    │   │   └── 单相机网格 (CameraDisplayWidget x8)
    │   └── 全尺寸拼接页面 (full_stitch_widget)
    └── 其他辅助组件
```

### 2. 数据流架构

```
camera_manager::FrameChannel
    ↓ (回调函数)
Widget::consumerThread()
    ↓ (GPU → CPU 转换)
AVFrame (CPU)
    ↓ (NV12 数据提取)
m_buffer (std::vector<uchar>)
    ↓ (OpenGL 渲染)
Nv12Render::render()
    ↓ (着色器处理)
屏幕显示
```

### 3. 渲染流程

```
1. consumerThread() 接收帧数据
2. av_hwframe_transfer_data() GPU → CPU 转换
3. 提取 Y 和 UV 数据到 m_buffer
4. paintGL() 调用 Nv12Render::render()
5. OpenGL 着色器将 NV12 转换为 RGB
6. 显示到屏幕
```

## 核心组件详解

### 1. StitchMainWindow（主窗口）

**职责**：
- 管理整个应用程序的主窗口界面
- 统一管理 camera_manager 的启动和停止
- 提供多页面布局（主页面和全尺寸拼接页面）
- 处理用户交互（鼠标点击切换页面）

**关键特性**：
- **单例 camera_manager**：统一管理，避免重复启动
- **QStackedLayout**：支持多页面切换
- **响应式布局**：使用 QVBoxLayout 和 QGridLayout 实现自适应布局
- **全屏显示**：默认全屏显示

**页面结构**：
1. **主页面（索引 0）**：
   - 上层：红外拼接显示区域（可配置）
   - 中层：可见光拼接显示区域（主要显示区域）
   - 下层：8 路单相机网格显示（2 行 4 列）

2. **全尺寸拼接页面（索引 1）**：
   - 全尺寸拼接图像显示（固定尺寸 20803x2160）
   - 支持滚动查看

**关键接口**：
```cpp
explicit StitchMainWindow(QWidget *parent = nullptr);
void setInfraredStitchWidget(QWidget* widget);  // 设置红外拼接组件
```

**用户交互**：
- **左键点击**：切换到主页面（索引 0）
- **右键点击**：切换到全尺寸拼接页面（索引 1）
- **关闭窗口**：自动停止 camera_manager

**使用示例**：
```cpp
StitchMainWindow window;
window.show();
// 或
window.showFullScreen();
```

### 2. Nv12Render（NV12 渲染器）

**职责**：
- 提供高效的 NV12 格式视频渲染
- 使用 OpenGL 着色器进行 YUV 到 RGB 转换
- 管理 OpenGL 纹理和缓冲区

**关键特性**：
- **硬件加速**：使用 OpenGL 进行渲染
- **着色器转换**：在 GPU 上完成 YUV 到 RGB 转换
- **步长支持**：支持自定义 Y 和 UV 步长
- **纹理管理**：自动管理 Y 和 UV 纹理

**渲染流程**：
1. `initialize()`：初始化着色器、几何体和纹理
2. `render()`：上传 Y 和 UV 数据到纹理
3. 执行着色器程序进行渲染

**关键接口**：
```cpp
bool initialize();  // 初始化渲染器
void render(uchar* nv12Ptr, int w, int h, int y_stride, int uv_stride);  // 渲染 NV12 数据
void render(uchar* y_data, uchar* uv_data, int w, int h, int y_stride, int uv_stride);  // 渲染分离的 Y/UV 数据
```

**着色器说明**：
- **顶点着色器**：处理顶点位置和纹理坐标
- **片段着色器**：将 YUV 转换为 RGB
  - YUV 到 RGB 转换矩阵
  - 支持 NV12 格式（Y 平面 + 交错 UV 平面）

**使用示例**：
```cpp
Nv12Render* render = new Nv12Render();
render->initialize();
render->render(y_data, uv_data, width, height, y_stride, uv_stride);
```

### 3. visible_camera_widget（可见光拼接显示组件）

**职责**：
- 显示可见光拼接视频流
- 实时更新性能指标（延迟时间）
- 通过信号发送标题更新

**关键特性**：
- **回调机制**：通过 `camera_manager::setStitchStreamCallback()` 接收帧数据
- **GPU → CPU 转换**：使用 `av_hwframe_transfer_data()` 转换 CUDA 帧
- **线程安全**：使用互斥锁保护缓冲区访问
- **性能监控**：计算并显示解码到拼接的延迟时间

**数据流**：
```
camera_manager::FrameChannel
    → consumerThread() (回调)
    → av_hwframe_transfer_data() (GPU → CPU)
    → 提取 NV12 数据到 m_buffer
    → paintGL() 渲染
```

**关键接口**：
```cpp
explicit visible_camera_widget(QWidget *parent = nullptr);
signals:
    void VisibleTitle(const QString& title);  // 发送标题更新信号
```

**性能监控**：
- 计算每个相机的解码到拼接延迟
- 计算平均延迟时间
- 通过信号更新标题显示延迟信息

### 4. full_stitch_widget（全尺寸拼接显示组件）

**职责**：
- 显示全尺寸拼接图像（固定尺寸 20803x2160）
- 支持滚动查看大尺寸图像
- 通过回调接收拼接帧数据

**关键特性**：
- **固定尺寸**：20803x2160 像素
- **回调机制**：通过 `camera_manager::setStitchStreamCallback()` 接收帧数据
- **内存对齐**：使用对齐内存分配优化性能
- **滚动支持**：嵌入 QScrollArea 支持滚动查看

**使用场景**：
- 查看完整尺寸的拼接图像
- 用于调试和验证拼接效果
- 支持放大查看细节

**关键接口**：
```cpp
explicit full_stitch_widget(QWidget *parent = nullptr);
```

**实现细节**：
- 使用 `aligned_alloc()` 进行内存对齐分配
- 支持自定义步长的 NV12 数据渲染
- 自动处理 GPU 到 CPU 的帧转换

### 5. CameraDisplayWidget（单相机显示组件）

**职责**：
- 显示单个相机的视频流
- 在网格布局中显示多个相机视图
- 支持相机名称标签显示

**关键特性**：
- **网格布局**：在 2 行 4 列的网格中显示
- **相机配置**：从 CameraConfig 读取相机信息
- **回调机制**：通过 `camera_manager::getSingleCameraSubStream()` 获取流
- **标签显示**：显示相机名称

**使用场景**：
- 在主页面底部显示 8 路单相机视图
- 用于监控单个相机的视频流
- 支持 `enable_view` 配置控制显示

**关键接口**：
```cpp
explicit CameraDisplayWidget(CameraConfig camera_config, QWidget *parent = nullptr);
```

**布局管理**：
- 每个相机占用一个网格单元
- 上方显示相机名称标签
- 下方显示视频画面
- 支持最多 8 个相机（2 行 4 列）

### 6. InfraredWidget（红外拼接显示组件）

**职责**：
- 显示红外拼接视频流
- 实时更新性能指标
- 通过信号发送标题更新

**关键特性**：
- **红外视频流**：显示红外相机的拼接结果
- **性能监控**：计算并显示延迟时间
- **可配置显示**：可通过 `setInfraredStitchWidget()` 动态设置
- **占位符支持**：无红外相机时显示黑色占位符

**使用场景**：
- 在主页面上层显示红外拼接图像
- 用于监控红外相机系统
- 与可见光拼接配合使用

**关键接口**：
```cpp
explicit InfraredWidget(QWidget *parent = nullptr);
signals:
    void IRTitle(const QString& title);  // 发送标题更新信号
```

### 7. widget_for_test（测试组件）

**职责**：
- 用于测试和调试的显示组件
- 支持指定 pipeline_id 和尺寸
- 可用于性能测试和功能验证

**关键特性**：
- **可配置参数**：支持指定 pipeline_id、宽度、高度
- **测试用途**：用于开发和调试
- **独立显示**：可独立使用进行测试

**关键接口**：
```cpp
explicit widget_for_test(int pipeline_id, int width, int height, QWidget *parent = nullptr);
```

## 技术实现细节

### 1. OpenGL 渲染流程

**初始化阶段**：
```cpp
void initializeGL() {
    // 1. 初始化 OpenGL 函数
    initializeOpenGLFunctions();
    
    // 2. 设置着色器
    setupShaders();
    
    // 3. 设置几何体（全屏四边形）
    setupGeometry();
    
    // 4. 创建纹理
    setupTextures();
}
```

**渲染阶段**：
```cpp
void paintGL() {
    // 1. 锁定互斥锁（如果使用多线程）
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // 2. 检查缓冲区有效性
    if (!m_buffer.empty() && m_width > 0 && m_height > 0) {
        // 3. 调用渲染器渲染
        m_render->render(m_buffer.data(), m_width, m_height, m_y_stride, m_uv_stride);
    }
}
```

### 2. GPU 到 CPU 帧转换

**转换流程**：
```cpp
void consumerThread(Frame frame) {
    AVFrame* src_frame = frame.m_data;
    AVFrame* process_frame = nullptr;
    
    // 1. 检查帧格式
    if (src_frame->format == AV_PIX_FMT_CUDA) {
        // 2. 转换 GPU 帧到 CPU
        if (av_hwframe_transfer_data(cpu_frame, src_frame, 0) < 0) {
            // 错误处理
            return;
        }
        process_frame = cpu_frame;
    }
    
    // 3. 提取 NV12 数据
    // 4. 更新缓冲区
    // 5. 触发重绘
    update();
}
```

### 3. 内存对齐分配

**对齐分配实现**：
```cpp
void* aligned_alloc(size_t size, size_t alignment) {
    void* ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = nullptr;
    }
#endif
    return ptr;
}
```

**使用场景**：
- NV12 数据缓冲区需要对齐以提高性能
- OpenGL 纹理上传需要对齐的数据
- SIMD 指令优化需要对齐的内存

### 4. 步长处理

**步长说明**：
- **Y 步长（y_stride）**：Y 平面每行的字节数（可能包含填充）
- **UV 步长（uv_stride）**：UV 平面每行的字节数（可能包含填充）
- **实际宽度**：图像的实际宽度（可能小于步长）

**处理方式**：
```cpp
// 提取 Y 数据
for (int y = 0; y < height; ++y) {
    memcpy(y_buffer + y * actual_y_stride, 
           src_y + y * src_y_stride, 
           width);
}

// 提取 UV 数据
for (int y = 0; y < height / 2; ++y) {
    memcpy(uv_buffer + y * actual_uv_stride, 
           src_uv + y * src_uv_stride, 
           width);
}
```

### 5. 线程安全

**互斥锁使用**：
```cpp
class visible_camera_widget {
private:
    std::mutex m_mutex;  // 保护缓冲区访问
    
    void consumerThread(Frame frame) {
        std::unique_lock<std::mutex> lock(m_mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            // 如果无法获取锁，跳过此帧
            return;
        }
        // 更新缓冲区...
    }
    
    void paintGL() {
        std::lock_guard<std::mutex> lock(m_mutex);
        // 读取缓冲区并渲染...
    }
};
```

## 配置和依赖

### 1. CMake 配置

**关键配置**：
```cmake
# Qt 自动处理
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# 包含目录
target_include_directories(qt PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../camera_manager/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/utils/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/shm/include
    ...
)

# 链接库
target_link_libraries(qt PRIVATE
    Qt5::Core
    Qt5::Widgets
    OpenGL::GL
    avcodec avformat avutil swscale
    pthread
    config
    camera_manager
    utils
)
```

### 2. 依赖库

**Qt 库**：
- `Qt5::Core`：Qt 核心功能
- `Qt5::Widgets`：Qt 窗口组件

**OpenGL**：
- `OpenGL::GL`：OpenGL 支持

**FFmpeg 库**：
- `avcodec`：编解码
- `avformat`：格式处理
- `avutil`：工具函数
- `swscale`：图像缩放

**项目内部库**：
- `camera_manager`：相机管理器
- `config`：配置管理
- `utils`：工具函数

## 使用流程

### 1. 基本使用

```cpp
#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    
    StitchMainWindow window;
    window.showFullScreen();
    
    return app.exec();
}
```

### 2. 自定义红外组件

```cpp
StitchMainWindow window;

// 创建自定义红外组件
InfraredWidget* customIRWidget = new InfraredWidget(&window);
window.setInfraredStitchWidget(customIRWidget);

window.show();
```

### 3. 访问相机流

```cpp
// 在主窗口内部
camera_manager* cam = camera_manager::GetInstance();

// 获取拼接流
FrameChannel* stitch_stream = cam->getStitchCameraStream(0);

// 获取单相机流
FrameChannel* single_stream = cam->getSingleCameraSubStream(0);
```

## 性能优化

### 1. OpenGL 渲染优化

- **纹理复用**：复用 Y 和 UV 纹理，避免频繁创建/销毁
- **着色器缓存**：使用 `addCacheableShaderFromSourceCode()` 缓存着色器
- **批量上传**：一次性上传完整的 Y 和 UV 数据
- **步长优化**：使用 `glPixelStorei(GL_UNPACK_ROW_LENGTH)` 处理步长

### 2. 内存管理优化

- **对齐分配**：使用对齐内存提高访问效率
- **缓冲区复用**：复用 `m_buffer` 避免频繁分配
- **及时释放**：及时释放不再使用的 AVFrame

### 3. 线程优化

- **非阻塞锁**：使用 `try_to_lock` 避免阻塞
- **最小锁范围**：只在必要时持有锁
- **帧跳过**：无法获取锁时跳过帧，避免延迟累积

### 4. GPU 转换优化

- **异步转换**：GPU 到 CPU 转换在独立线程中进行
- **批量处理**：批量处理多帧数据
- **错误处理**：转换失败时及时释放资源

## 常见问题

### 1. 黑屏问题

**可能原因**：
- OpenGL 上下文未正确初始化
- 缓冲区数据为空
- 渲染器未正确初始化

**解决方法**：
- 检查 `initializeGL()` 是否被调用
- 检查 `m_buffer` 是否有数据
- 检查 `m_render->initialize()` 返回值
- 检查 OpenGL 上下文是否有效

### 2. 画面卡顿

**可能原因**：
- 帧率过高，渲染跟不上
- 互斥锁竞争激烈
- GPU 转换耗时过长

**解决方法**：
- 降低帧率或跳过部分帧
- 优化锁的使用范围
- 检查 GPU 转换性能
- 使用性能分析工具定位瓶颈

### 3. 内存泄漏

**可能原因**：
- AVFrame 未正确释放
- OpenGL 资源未正确清理
- 缓冲区未及时释放

**解决方法**：
- 确保所有 `av_frame_alloc()` 都有对应的 `av_frame_free()`
- 在析构函数中清理 OpenGL 资源
- 检查缓冲区生命周期

### 4. 画面撕裂

**可能原因**：
- 垂直同步未启用
- 渲染和更新不同步

**解决方法**：
- 启用垂直同步（VSync）
- 使用双缓冲
- 同步渲染和更新时机

### 5. 步长问题

**可能原因**：
- Y 和 UV 步长计算错误
- 数据对齐问题

**解决方法**：
- 检查步长计算逻辑
- 使用对齐内存分配
- 验证数据提取正确性

## 扩展开发

### 1. 添加新的显示组件

1. 继承 `QOpenGLWidget`
2. 实现 `initializeGL()`, `paintGL()`, `resizeGL()`
3. 实现 `consumerThread()` 接收帧数据
4. 使用 `Nv12Render` 进行渲染

**示例**：
```cpp
class MyCustomWidget : public QOpenGLWidget {
    Q_OBJECT
public:
    explicit MyCustomWidget(QWidget *parent = nullptr);
    
protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;
    void consumerThread(Frame frame);
    
private:
    Nv12Render* m_render;
    std::vector<uchar> m_buffer;
    // ...
};
```

### 2. 自定义渲染器

1. 继承或修改 `Nv12Render`
2. 实现自定义着色器
3. 添加新的渲染方法

### 3. 添加新的页面

1. 在 `StitchMainWindow` 中添加新的页面
2. 使用 `QStackedLayout` 管理页面
3. 添加页面切换逻辑

### 4. 性能监控扩展

1. 添加更多性能指标
2. 实现性能数据可视化
3. 添加性能日志记录

## 调试技巧

### 1. OpenGL 调试

- 使用 `glGetError()` 检查 OpenGL 错误
- 使用 Qt Creator 的 OpenGL 调试工具
- 检查着色器编译错误

### 2. 帧数据调试

- 打印帧尺寸和格式信息
- 检查缓冲区数据有效性
- 验证步长计算正确性

### 3. 性能分析

- 使用 Qt Creator 的性能分析器
- 测量各阶段耗时
- 监控内存使用情况

### 4. 日志输出

- 使用 `qDebug()` 输出调试信息
- 使用 `LOG_DEBUG` 输出详细日志
- 检查关键函数的执行流程


