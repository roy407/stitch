# Components Module Documentation

## Overview

The `components` module provides Qt-based graphical user interface components for displaying and processing multi-camera stitched video streams. This module contains multiple OpenGL rendering components that support real-time display of visible light stitching, infrared stitching, single camera views, and provides complete main window interface management.

## Core Features

- **OpenGL Hardware-Accelerated Rendering**: High-performance video rendering based on QOpenGLWidget
- **NV12 Format Support**: Efficient YUV420SP format video rendering
- **Multi-View Display**: Support various display modes including visible light stitching, infrared stitching, single camera views
- **Real-time Performance Monitoring**: Display frame rate, latency and other performance metrics
- **Responsive Layout**: Support window scaling and adaptive layout
- **Full-screen Display**: Support full-screen mode and window mode switching

## Directory Structure

```
components/
├── CMakeLists.txt          # Component build configuration
└── qt/                     # Qt components directory
    ├── CMakeLists.txt      # Qt component build configuration
    ├── include/            # Header files directory
    │   ├── mainwindow.h              # Main window class
    │   ├── full_stitch_widget.h      # Full-size stitching display component
    │   ├── visible_camera_widget.h   # Visible light stitching display component
    │   ├── visible_single_camera_widget.h  # Single camera display component
    │   ├── infrared_camera_widget.h  # Infrared stitching display component
    │   ├── nv12render.h              # NV12 format OpenGL renderer
    │   └── widget_for_test.h        # Test component
    └── src/                # Source files directory
        ├── mainwindow.cpp
        ├── full_stitch_widget.cpp
        ├── visible_camera_widget.cpp
        ├── visible_single_camera_widget.cpp
        ├── infrared_camera_widget.cpp
        ├── nv12render.cpp
        └── widget_for_test.cpp
```

## Architecture Design

### 1. Component Hierarchy

```
QMainWindow
└── StitchMainWindow
    ├── QStackedLayout (Multi-page layout)
    │   ├── Main Page (mainWidget)
    │   │   ├── Infrared Stitching Area (InfraredWidget)
    │   │   ├── Visible Light Stitching Area (visible_camera_widget)
    │   │   └── Single Camera Grid (CameraDisplayWidget x8)
    │   └── Full-size Stitching Page (full_stitch_widget)
    └── Other auxiliary components
```

### 2. Data Flow Architecture

```
camera_manager::FrameChannel
    ↓ (Callback function)
Widget::consumerThread()
    ↓ (GPU → CPU conversion)
AVFrame (CPU)
    ↓ (NV12 data extraction)
m_buffer (std::vector<uchar>)
    ↓ (OpenGL rendering)
Nv12Render::render()
    ↓ (Shader processing)
Screen Display
```

### 3. Rendering Flow

```
1. consumerThread() receives frame data
2. av_hwframe_transfer_data() GPU → CPU conversion
3. Extract Y and UV data to m_buffer
4. paintGL() calls Nv12Render::render()
5. OpenGL shader converts NV12 to RGB
6. Display to screen
```

## Core Components

### 1. StitchMainWindow (Main Window)

**Responsibilities**:
- Manage the main window interface of the entire application
- Unified management of camera_manager startup and shutdown
- Provide multi-page layout (main page and full-size stitching page)
- Handle user interactions (mouse clicks to switch pages)

**Key Features**:
- **Singleton camera_manager**: Unified management, avoid duplicate startup
- **QStackedLayout**: Support multi-page switching
- **Responsive Layout**: Use QVBoxLayout and QGridLayout for adaptive layout
- **Full-screen Display**: Default full-screen display

**Page Structure**:
1. **Main Page (Index 0)**:
   - Upper: Infrared stitching display area (configurable)
   - Middle: Visible light stitching display area (main display area)
   - Lower: 8-way single camera grid display (2 rows 4 columns)

2. **Full-size Stitching Page (Index 1)**:
   - Full-size stitched image display (fixed size 20803x2160)
   - Support scrolling view

**Key Interfaces**:
```cpp
explicit StitchMainWindow(QWidget *parent = nullptr);
void setInfraredStitchWidget(QWidget* widget);  // Set infrared stitching component
```

**User Interactions**:
- **Left Click**: Switch to main page (index 0)
- **Right Click**: Switch to full-size stitching page (index 1)
- **Close Window**: Automatically stop camera_manager

**Usage Example**:
```cpp
StitchMainWindow window;
window.show();
// or
window.showFullScreen();
```

### 2. Nv12Render (NV12 Renderer)

**Responsibilities**:
- Provide efficient NV12 format video rendering
- Use OpenGL shaders for YUV to RGB conversion
- Manage OpenGL textures and buffers

**Key Features**:
- **Hardware Acceleration**: Use OpenGL for rendering
- **Shader Conversion**: YUV to RGB conversion done on GPU
- **Stride Support**: Support custom Y and UV strides
- **Texture Management**: Automatically manage Y and UV textures

**Rendering Flow**:
1. `initialize()`: Initialize shaders, geometry and textures
2. `render()`: Upload Y and UV data to textures
3. Execute shader program for rendering

**Key Interfaces**:
```cpp
bool initialize();  // Initialize renderer
void render(uchar* nv12Ptr, int w, int h, int y_stride, int uv_stride);  // Render NV12 data
void render(uchar* y_data, uchar* uv_data, int w, int h, int y_stride, int uv_stride);  // Render separated Y/UV data
```

**Shader Description**:
- **Vertex Shader**: Handle vertex positions and texture coordinates
- **Fragment Shader**: Convert YUV to RGB
  - YUV to RGB conversion matrix
  - Support NV12 format (Y plane + interleaved UV plane)

**Usage Example**:
```cpp
Nv12Render* render = new Nv12Render();
render->initialize();
render->render(y_data, uv_data, width, height, y_stride, uv_stride);
```

### 3. visible_camera_widget (Visible Light Stitching Display Component)

**Responsibilities**:
- Display visible light stitched video stream
- Real-time update of performance metrics (latency time)
- Send title updates through signals

**Key Features**:
- **Callback Mechanism**: Receive frame data through `camera_manager::setStitchStreamCallback()`
- **GPU → CPU Conversion**: Use `av_hwframe_transfer_data()` to convert CUDA frames
- **Thread Safety**: Use mutex to protect buffer access
- **Performance Monitoring**: Calculate and display latency from decoding to stitching

**Data Flow**:
```
camera_manager::FrameChannel
    → consumerThread() (callback)
    → av_hwframe_transfer_data() (GPU → CPU)
    → Extract NV12 data to m_buffer
    → paintGL() rendering
```

**Key Interfaces**:
```cpp
explicit visible_camera_widget(QWidget *parent = nullptr);
signals:
    void VisibleTitle(const QString& title);  // Send title update signal
```

**Performance Monitoring**:
- Calculate decoding to stitching latency for each camera
- Calculate average latency time
- Update title display latency information through signals

### 4. full_stitch_widget (Full-size Stitching Display Component)

**Responsibilities**:
- Display full-size stitched image (fixed size 20803x2160)
- Support scrolling view for large-size images
- Receive stitched frame data through callback

**Key Features**:
- **Fixed Size**: 20803x2160 pixels
- **Callback Mechanism**: Receive frame data through `camera_manager::setStitchStreamCallback()`
- **Memory Alignment**: Use aligned memory allocation for performance optimization
- **Scrolling Support**: Embedded QScrollArea supports scrolling view

**Use Cases**:
- View full-size stitched images
- Used for debugging and verifying stitching effects
- Support zooming to view details

**Key Interfaces**:
```cpp
explicit full_stitch_widget(QWidget *parent = nullptr);
```

**Implementation Details**:
- Use `aligned_alloc()` for aligned memory allocation
- Support custom stride NV12 data rendering
- Automatically handle GPU to CPU frame conversion

### 5. CameraDisplayWidget (Single Camera Display Component)

**Responsibilities**:
- Display single camera video stream
- Display multiple camera views in grid layout
- Support camera name label display

**Key Features**:
- **Grid Layout**: Display in 2 rows 4 columns grid
- **Camera Configuration**: Read camera information from CameraConfig
- **Callback Mechanism**: Get stream through `camera_manager::getSingleCameraSubStream()`
- **Label Display**: Display camera name

**Use Cases**:
- Display 8-way single camera views at bottom of main page
- Used for monitoring single camera video streams
- Support `enable_view` configuration control display

**Key Interfaces**:
```cpp
explicit CameraDisplayWidget(CameraConfig camera_config, QWidget *parent = nullptr);
```

**Layout Management**:
- Each camera occupies one grid cell
- Display camera name label above
- Display video frame below
- Support up to 8 cameras (2 rows 4 columns)

### 6. InfraredWidget (Infrared Stitching Display Component)

**Responsibilities**:
- Display infrared stitched video stream
- Real-time update of performance metrics
- Send title updates through signals

**Key Features**:
- **Infrared Video Stream**: Display infrared camera stitching results
- **Performance Monitoring**: Calculate and display latency time
- **Configurable Display**: Can be dynamically set through `setInfraredStitchWidget()`
- **Placeholder Support**: Display black placeholder when no infrared camera

**Use Cases**:
- Display infrared stitched image at upper layer of main page
- Used for monitoring infrared camera systems
- Used together with visible light stitching

**Key Interfaces**:
```cpp
explicit InfraredWidget(QWidget *parent = nullptr);
signals:
    void IRTitle(const QString& title);  // Send title update signal
```

### 7. widget_for_test (Test Component)

**Responsibilities**:
- Display component for testing and debugging
- Support specified pipeline_id and size
- Can be used for performance testing and function verification

**Key Features**:
- **Configurable Parameters**: Support specified pipeline_id, width, height
- **Testing Purpose**: Used for development and debugging
- **Independent Display**: Can be used independently for testing

**Key Interfaces**:
```cpp
explicit widget_for_test(int pipeline_id, int width, int height, QWidget *parent = nullptr);
```

## Technical Implementation Details

### 1. OpenGL Rendering Flow

**Initialization Phase**:
```cpp
void initializeGL() {
    // 1. Initialize OpenGL functions
    initializeOpenGLFunctions();
    
    // 2. Setup shaders
    setupShaders();
    
    // 3. Setup geometry (full-screen quadrilateral)
    setupGeometry();
    
    // 4. Create textures
    setupTextures();
}
```

**Rendering Phase**:
```cpp
void paintGL() {
    // 1. Lock mutex (if using multi-threading)
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // 2. Check buffer validity
    if (!m_buffer.empty() && m_width > 0 && m_height > 0) {
        // 3. Call renderer to render
        m_render->render(m_buffer.data(), m_width, m_height, m_y_stride, m_uv_stride);
    }
}
```

### 2. GPU to CPU Frame Conversion

**Conversion Flow**:
```cpp
void consumerThread(Frame frame) {
    AVFrame* src_frame = frame.m_data;
    AVFrame* process_frame = nullptr;
    
    // 1. Check frame format
    if (src_frame->format == AV_PIX_FMT_CUDA) {
        // 2. Convert GPU frame to CPU
        if (av_hwframe_transfer_data(cpu_frame, src_frame, 0) < 0) {
            // Error handling
            return;
        }
        process_frame = cpu_frame;
    }
    
    // 3. Extract NV12 data
    // 4. Update buffer
    // 5. Trigger repaint
    update();
}
```

### 3. Memory Alignment Allocation

**Aligned Allocation Implementation**:
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

**Use Cases**:
- NV12 data buffer needs alignment for performance
- OpenGL texture upload requires aligned data
- SIMD instruction optimization requires aligned memory

### 4. Stride Processing

**Stride Description**:
- **Y Stride (y_stride)**: Bytes per row of Y plane (may include padding)
- **UV Stride (uv_stride)**: Bytes per row of UV plane (may include padding)
- **Actual Width**: Actual width of image (may be less than stride)

**Processing Method**:
```cpp
// Extract Y data
for (int y = 0; y < height; ++y) {
    memcpy(y_buffer + y * actual_y_stride, 
           src_y + y * src_y_stride, 
           width);
}

// Extract UV data
for (int y = 0; y < height / 2; ++y) {
    memcpy(uv_buffer + y * actual_uv_stride, 
           src_uv + y * src_uv_stride, 
           width);
}
```

### 5. Thread Safety

**Mutex Usage**:
```cpp
class visible_camera_widget {
private:
    std::mutex m_mutex;  // Protect buffer access
    
    void consumerThread(Frame frame) {
        std::unique_lock<std::mutex> lock(m_mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            // If cannot acquire lock, skip this frame
            return;
        }
        // Update buffer...
    }
    
    void paintGL() {
        std::lock_guard<std::mutex> lock(m_mutex);
        // Read buffer and render...
    }
};
```

## Configuration and Dependencies

### 1. CMake Configuration

**Key Configuration**:
```cmake
# Qt auto processing
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Include directories
target_include_directories(qt PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../camera_manager/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/utils/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/shm/include
    ...
)

# Link libraries
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

### 2. Dependencies

**Qt Libraries**:
- `Qt5::Core`: Qt core functionality
- `Qt5::Widgets`: Qt window components

**OpenGL**:
- `OpenGL::GL`: OpenGL support

**FFmpeg Libraries**:
- `avcodec`: Codec
- `avformat`: Format handling
- `avutil`: Utility functions
- `swscale`: Image scaling

**Internal Project Libraries**:
- `camera_manager`: Camera manager
- `config`: Configuration management
- `utils`: Utility functions

## Usage Flow

### 1. Basic Usage

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

### 2. Custom Infrared Component

```cpp
StitchMainWindow window;

// Create custom infrared component
InfraredWidget* customIRWidget = new InfraredWidget(&window);
window.setInfraredStitchWidget(customIRWidget);

window.show();
```

### 3. Access Camera Streams

```cpp
// Inside main window
camera_manager* cam = camera_manager::GetInstance();

// Get stitched stream
FrameChannel* stitch_stream = cam->getStitchCameraStream(0);

// Get single camera stream
FrameChannel* single_stream = cam->getSingleCameraSubStream(0);
```

## Performance Optimization

### 1. OpenGL Rendering Optimization

- **Texture Reuse**: Reuse Y and UV textures, avoid frequent create/destroy
- **Shader Caching**: Use `addCacheableShaderFromSourceCode()` to cache shaders
- **Batch Upload**: Upload complete Y and UV data at once
- **Stride Optimization**: Use `glPixelStorei(GL_UNPACK_ROW_LENGTH)` to handle stride

### 2. Memory Management Optimization

- **Aligned Allocation**: Use aligned memory for access efficiency
- **Buffer Reuse**: Reuse `m_buffer` to avoid frequent allocation
- **Timely Release**: Release unused AVFrames promptly

### 3. Thread Optimization

- **Non-blocking Lock**: Use `try_to_lock` to avoid blocking
- **Minimal Lock Scope**: Hold lock only when necessary
- **Frame Skipping**: Skip frames when cannot acquire lock, avoid latency accumulation

### 4. GPU Conversion Optimization

- **Asynchronous Conversion**: GPU to CPU conversion in independent thread
- **Batch Processing**: Batch process multiple frames
- **Error Handling**: Release resources promptly on conversion failure

## Common Issues

### 1. Black Screen

**Possible Causes**:
- OpenGL context not properly initialized
- Buffer data is empty
- Renderer not properly initialized

**Solutions**:
- Check if `initializeGL()` is called
- Check if `m_buffer` has data
- Check if `m_render->initialize()` returns true
- Check if OpenGL context is valid

### 2. Frame Stuttering

**Possible Causes**:
- Frame rate too high, rendering cannot keep up
- Mutex contention is intense
- GPU conversion takes too long

**Solutions**:
- Reduce frame rate or skip some frames
- Optimize lock usage scope
- Check GPU conversion performance
- Use profiling tools to locate bottlenecks

### 3. Memory Leaks

**Possible Causes**:
- AVFrame not properly released
- OpenGL resources not properly cleaned up
- Buffers not released promptly

**Solutions**:
- Ensure all `av_frame_alloc()` have corresponding `av_frame_free()`
- Clean up OpenGL resources in destructor
- Check buffer lifecycle

### 4. Screen Tearing

**Possible Causes**:
- Vertical sync not enabled
- Rendering and update not synchronized

**Solutions**:
- Enable vertical sync (VSync)
- Use double buffering
- Synchronize rendering and update timing

### 5. Stride Issues

**Possible Causes**:
- Y and UV stride calculation error
- Data alignment issues

**Solutions**:
- Check stride calculation logic
- Use aligned memory allocation
- Verify data extraction correctness

## Extension Development

### 1. Adding New Display Components

1. Inherit from `QOpenGLWidget`
2. Implement `initializeGL()`, `paintGL()`, `resizeGL()`
3. Implement `consumerThread()` to receive frame data
4. Use `Nv12Render` for rendering

**Example**:
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

### 2. Custom Renderer

1. Inherit or modify `Nv12Render`
2. Implement custom shaders
3. Add new rendering methods

### 3. Adding New Pages

1. Add new page in `StitchMainWindow`
2. Use `QStackedLayout` to manage pages
3. Add page switching logic

### 4. Performance Monitoring Extension

1. Add more performance metrics
2. Implement performance data visualization
3. Add performance log recording

## Debugging Tips

### 1. OpenGL Debugging

- Use `glGetError()` to check OpenGL errors
- Use Qt Creator's OpenGL debugging tools
- Check shader compilation errors

### 2. Frame Data Debugging

- Print frame size and format information
- Check buffer data validity
- Verify stride calculation correctness

### 3. Performance Analysis

- Use Qt Creator's profiler
- Measure time consumption at each stage
- Monitor memory usage

### 4. Log Output

- Use `qDebug()` to output debug information
- Use `LOG_DEBUG` to output detailed logs
- Check execution flow of key functions
