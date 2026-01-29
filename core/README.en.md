# Core Module Documentation

## Overview

The `core` module is the core foundation module of the project, providing basic functionalities such as configuration management, CUDA operators, and utility functions. This module provides configuration management, high-performance CUDA kernel operations, and general utility support for the entire system.

## Core Features

- **Configuration Management**: Flexible JSON-based configuration system
- **CUDA Acceleration**: High-performance GPU image processing kernels
- **Utility Functions**: Common functions such as logging, queues, and utility functions
- **Thread Safety**: Provide thread-safe data structures

## Directory Structure

```
core/
├── CMakeLists.txt          # Core module build configuration
├── README.md              # This document
├── config/                # Configuration management module
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── config.h       # Configuration management class
│   └── src/
│       └── config.cpp     # Configuration implementation
├── operator/              # CUDA operator module
│   ├── CMakeLists.txt
│   ├── README.md
│   ├── include/
│   │   ├── nvidia_kernel.h  # CUDA kernel header file summary
│   │   └── nvidia/          # CUDA kernel header files directory
│   │       ├── stitch_with_mapping_table.cuh      # Mapping table stitching
│   │       ├── stitch_with_mapping_table_yuv420p.cuh
│   │       ├── stitch_raw.cuh                     # Raw stitching
│   │       ├── stitch_raw_yuv420p.cuh
│   │       ├── stitch_with_crop.cuh               # Crop stitching
│   │       ├── stitch_with_h_matrix.cuh            # Homography matrix stitching
│   │       ├── h_matrix_inv/                      # Inverse homography matrix stitching
│   │       │   ├── h_matrix_inv.h
│   │       │   ├── stitch_with_h_matrix_inv.cuh
│   │       │   ├── stitch_with_h_matrix_inv_v1.1.cuh
│   │       │   └── stitch_with_h_matrix_inv_v2.cuh
│   │       ├── resize.cuh                         # Image resizing
│   │       ├── scale.cuh                          # Image scaling
│   │       └── yuv_2_rgb.cuh                      # YUV to RGB
│   └── src/
│       └── nvidia/        # CUDA kernel implementation
│           ├── stitch_with_mapping_table.cu
│           ├── stitch_raw.cu
│           ├── resize.cu
│           └── ...
└── utils/                 # Utility functions module
    ├── CMakeLists.txt
    ├── include/
    │   ├── log.hpp        # Logging system
    │   ├── tools.hpp      # Utility functions
    │   ├── safe_queue.hpp # Thread-safe queue
    │   └── safe_list.h    # Thread-safe list (incomplete)
    └── src/
        └── tools.cpp      # Utility functions implementation
```

## Module Details

### 1. config Module (Configuration Management)

#### Overview

The `config` module provides a JSON-based configuration management system that supports loading camera configurations, stitching configurations, global configurations, and other information from configuration files. Uses singleton pattern to provide global configuration access.

#### Core Data Structures

**CameraConfig** (Camera Configuration):
```cpp
struct CameraConfig {
    std::string name;           // Camera name
    int cam_id;                 // Camera ID
    bool enable;                // Whether enabled
    std::string input_url;      // Input URL (RTSP/file path/USB device)
    int width, height;          // Resolution
    std::string output_url;     // Output URL
    bool enable_view;           // Whether to enable single view
    double scale_factor;        // Scale factor
    bool rtsp;                  // Whether to enable RTSP streaming
};
```

**MappingTableConfig** (Mapping Table Configuration):
```cpp
struct MappingTableConfig {
    std::string file_path;              // Mapping table file path
    cudaTextureObject_t d_mapping_table; // CUDA texture object
    int output_width;                   // Output width
};
```

**StitchConfig** (Stitching Configuration):
```cpp
struct StitchConfig {
    std::string stitch_mode;    // Stitching mode ("mapping_table", "raw", etc.)
    StitchImplConfig stitch_impl; // Stitching implementation configuration
    std::string output_url;      // Output URL
    double scale_factor;        // Scale factor
    bool rtsp;                   // Whether to enable RTSP streaming
};
```

**PipelineConfig** (Pipeline Configuration):
```cpp
struct PipelineConfig {
    std::string name;            // Pipeline name
    int pipeline_id;             // Pipeline ID
    bool enable;                 // Whether enabled
    bool use_substream;          // Whether to use substream
    uint64_t default_width;      // Default width
    uint64_t default_height;     // Default height
    std::string main_stream;      // Main stream file
    std::string sub_stream;       // Substream file
    std::vector<CameraConfig> cameras; // Camera list
    StitchConfig stitch;         // Stitching configuration
    bool openTimingWatcher;      // Whether to enable timing monitoring
};
```

**GlobalConfig** (Global Configuration):
```cpp
struct GlobalConfig {
    std::string loglevel;        // Log level
    std::string type;            // Input type ("rtsp", "mp4", "usb")
    std::string format;          // Pixel format ("YUV420", "YUV420P")
    int record_duration;         // Recording duration
    std::string record_path;     // Recording path
    std::string decoder;         // Decoder name
    std::string encoder;         // Encoder name
};
```

#### Core Interfaces

```cpp
class config {
public:
    // Set configuration file name (must be called before initialization)
    static void SetConfigFileName(std::string cfg_name);
    
    // Get configuration file name
    static std::string GetConfigFileName();
    
    // Get singleton instance
    static config& GetInstance();
    
    // Get complete configuration
    const Config GetConfig() const;
    
    // Get global configuration
    const GlobalConfig GetGlobalConfig() const;
    
    // Get specified pipeline configuration
    const PipelineConfig GetPipelineConfig(int pipeline_id) const;
    
    // Get specified pipeline camera configuration list
    const std::vector<CameraConfig> GetCamerasConfig(int pipeline_id) const;
    
    // Get specified pipeline stitching configuration
    const StitchConfig GetStitchConfig(int pipeline_id) const;
};
```

#### Usage Example

```cpp
// 1. Set configuration file name (at program startup)
config::SetConfigFileName("config.json");

// 2. Get configuration instance
auto& cfg = config::GetInstance();

// 3. Get global configuration
const GlobalConfig& global = cfg.GetGlobalConfig();
std::string decoder = global.decoder;

// 4. Get pipeline configuration
const PipelineConfig& pipeline = cfg.GetPipelineConfig(0);

// 5. Get camera configuration
const std::vector<CameraConfig>& cameras = cfg.GetCamerasConfig(0);

// 6. Use macro definition to simplify access
#define CFG_HANDLE config::GetInstance()
auto& pipeline = CFG_HANDLE.GetPipelineConfig(0);
```

#### Configuration File Format

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

#### Mapping Table Loading

The configuration system supports loading CUDA texture mapping tables from files:

```cpp
bool loadMappingTable(cudaTextureObject_t& tex,
                     const std::string filename,
                     uint64_t width,
                     uint64_t height);
```

The mapping table file format is a binary file containing an array of `MapEntry` structures:
```cpp
struct MapEntry {
    uint16_t cam_id;  // Camera ID
    uint16_t map_x;   // Mapping X coordinate
    uint16_t map_y;   // Mapping Y coordinate
    uint16_t pad;     // Alignment padding
};
```

### 2. operator Module (CUDA Operators)

#### Overview

The `operator` module provides high-performance CUDA-based image processing kernels, including various stitching algorithms, image scaling, format conversion, and other functions. All operations are executed on the GPU, providing extremely high processing performance.

#### Supported Stitching Kernels

**1. Mapping Table Stitching**

The most commonly used stitching method, using pre-computed mapping tables for stitching.

- `stitch_with_mapping_table.cuh`: NV12 format (YUV420SP)
- `stitch_with_mapping_table_yuv420p.cuh`: YUV420P format

**Interface**:
```cpp
void launch_stitch_kernel_with_mapping_table(
    uint8_t** inputs_y, uint8_t** inputs_uv,      // Input Y and UV data pointer arrays
    int* input_linesize_y, int* input_linesize_uv, // Input stride arrays
    uint8_t* output_y, uint8_t* output_uv,        // Output Y and UV data pointers
    int output_linesize_y, int output_linesize_uv, // Output strides
    int cam_num,                                   // Camera count
    int single_width,                              // Single camera width
    int width, int height,                         // Output size
    const cudaTextureObject_t mapping_table,       // CUDA texture mapping table
    cudaStream_t stream1, cudaStream_t stream2);  // CUDA streams (Y and UV parallel processing)
```

**Features**:
- Use CUDA texture memory to accelerate lookups
- Y and UV planes processed in parallel
- Support multi-camera stitching

**2. Raw Stitching**

Simple raw stitching method.

- `stitch_raw.cuh`: NV12 format
- `stitch_raw_yuv420p.cuh`: YUV420P format

**3. Crop Stitching**

Supports stitching with crop regions.

- `stitch_with_crop.cuh`

**4. Homography Matrix Stitching**

Stitching based on homography matrix.

- `stitch_with_h_matrix.cuh`

**5. Inverse Homography Matrix Stitching**

Stitching based on inverse homography matrix, supports multiple versions.

- `stitch_with_h_matrix_inv.cuh`: Base version
- `stitch_with_h_matrix_inv_v1.1.cuh`: Version 1.1
- `stitch_with_h_matrix_inv_v2.cuh`: Version 2

#### Image Processing Kernels

**1. Image Resizing**

```cpp
void ReSize(
    const uint8_t* pInYData, const uint8_t* pInUVData,  // Input Y and UV data
    int pInWidth, int pInHeight,                        // Input size
    int pInYStride, int pInUVStride,                    // Input strides
    uint8_t* pOutYData, uint8_t* pOutUVData,            // Output Y and UV data
    int pOutWidth, int pOutHeight,                       // Output size
    int pOutYStride, int pOutUVStride,                  // Output strides
    cudaStream_t stream);                               // CUDA stream
```

**2. Image Scaling**

Another scaling implementation.

**3. YUV to RGB**

```cpp
// Defined in yuv_2_rgb.cuh
```

#### Usage Example

```cpp
#include "nvidia_kernel.h"

// 1. Mapping table stitching
launch_stitch_kernel_with_mapping_table(
    inputs_y, inputs_uv,
    input_linesize_y, input_linesize_uv,
    output_y, output_uv,
    output_linesize_y, output_linesize_uv,
    cam_num, single_width, width, height,
    mapping_table,
    stream1, stream2
);

// 2. Image resizing
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

#### Build Configuration

```cmake
enable_language(CUDA)

add_library(operator_nvidia STATIC
  src/nvidia/scale.cu
  src/nvidia/stitch_raw.cu
  src/nvidia/stitch_with_mapping_table.cu
  # ... other CUDA source files
)

target_link_libraries(operator_nvidia
    avcodec avformat avutil swscale
    cuda nppicc nppc nppidei config
)
```

**Dependencies**:
- `cuda`: CUDA runtime
- `nppicc`, `nppc`, `nppidei`: NVIDIA Performance Primitives Library (NPP)
- FFmpeg libraries: For frame format processing

### 3. utils Module (Utility Functions)

#### Overview

The `utils` module provides common utility functions, including logging system, thread-safe queues, time utilities, frame processing utilities, etc.

#### Logging System (log.hpp)

High-performance logging system based on `spdlog`.

**Log Levels**:
- `LOG_DEBUG`: Debug information
- `LOG_INFO`: General information
- `LOG_WARN`: Warning information
- `LOG_ERROR`: Error information

**Usage Example**:
```cpp
#include "log.hpp"

LOG_DEBUG("Debug message: {}", value);
LOG_INFO("Info message: {}", value);
LOG_WARN("Warning message: {}", value);
LOG_ERROR("Error message: {}", value);
```

**Log Format**:
```
[2024-01-19 17:08:32.510][console][info][Pipeline.cpp:18][pid:153257] pipeline id : 0
```

**Configuration**:
- Log level read from configuration file (`global.loglevel`)
- Supported: `debug`, `info`, `warn`, `error`, `critical`

**Debug Macros**:
```cpp
// CUDA error checking
CHECK_CUDA(cudaMalloc(&ptr, size));

// Null pointer checking
CHECK_NULL(ptr);
CHECK_NULL_RETURN(ptr);
CHECK_NULL_RETURN_NULL(ptr);

// FFmpeg error checking
CHECK_FFMPEG_RETURN(av_frame_get_buffer(frame, 32));
CHECK_FFMPEG_RETURN_FUNC(ret, av_frame_get_buffer);
```

#### Thread-Safe Queue (safe_queue.hpp)

Templated thread-safe queue supporting blocking and non-blocking operations.

**Interface**:
```cpp
template<typename T>
class safe_queue {
public:
    void push(const T& value);           // Push element
    bool try_pop(T& result);              // Non-blocking pop
    bool wait_and_pop(T& result);        // Blocking pop
    bool wait_and_front(T& result);       // Blocking front
    bool empty() const;                   // Check if empty
    int size() const;                    // Queue size
    void pop_and_free();                 // Pop and free (for AVFrame/AVPacket)
    void clear();                        // Clear queue
    void stop();                         // Stop queue
    
    // Statistics
    int frames{0};      // Frame count
    int packets{0};     // Packet count
    int frame_lost{0};  // Lost frame count
    int packet_lost{0}; // Lost packet count
};
```

**Features**:
- Thread-safe: Uses mutex and condition variable
- Auto-release: Supports automatic release for `Packet` and `Frame` types
- Overflow protection: Automatically discards oldest elements when queue is full
- Stop mechanism: Supports graceful stop

**Usage Example**:
```cpp
#include "safe_queue.hpp"

safe_queue<Frame> frame_queue;

// Producer
Frame frame;
// ... fill frame ...
frame_queue.push(frame);

// Consumer (blocking)
Frame received;
while (frame_queue.wait_and_pop(received)) {
    // Process frame data
    processFrame(received);
    av_frame_free(&received.m_data);
}

// Consumer (non-blocking)
Frame received;
if (frame_queue.try_pop(received)) {
    // Process frame data
}
```

#### Utility Functions (tools.hpp)

**Time Utilities**:
```cpp
// Get current nanosecond timestamp
uint64_t get_now_time();

// Generate filename with timestamp
std::string get_current_time_filename(const std::string& suffix = ".txt");
```

**Frame Processing Utilities**:
```cpp
// String to AVPixelFormat
AVPixelFormat transfer_string_2_AVPixelFormat(std::string format);

// Save NV12 frame to file
void save_frame_as_nv12(AVFrame* frame, const std::string& filename);

// Convert CUDA frame to CPU and save as NV12
void transfer_and_save_cuda_nv12(AVFrame* hw_frame, const std::string& filename);

// Create frame in CPU memory
AVFrame* get_frame_on_cpu_memory(std::string format, int width, int height);

// Create frame in GPU memory
AVFrame* get_frame_on_gpu_memory(std::string format, int width, int height, AVBufferRef* av_buffer);
```

**Performance Statistics Utilities**:
```cpp
// Save timestamp to file
void save_cost_times_to_timestamped_file(const costTimes& t, std::ofstream& ofs);

// Save performance table as CSV
void save_cost_table_csv(const costTimes& t, std::ofstream& ofs);

// Print performance time
void printCostTimes(const costTimes& c);
```

**Drawing Utilities**:
```cpp
// Draw vertical line on NV12 frame
void draw_vertical_line_nv12(AVFrame *frame, int x, const std::string label, int fst, int Y);
```

**Usage Example**:
```cpp
#include "tools.hpp"

// Get current time
uint64_t now = get_now_time();

// Save frame
save_frame_as_nv12(frame, "output.nv12");

// Convert and save CUDA frame
transfer_and_save_cuda_nv12(cuda_frame, "output.nv12");

// Create CPU frame
AVFrame* cpu_frame = get_frame_on_cpu_memory("YUV420", 1920, 1080);

// Create GPU frame
AVFrame* gpu_frame = get_frame_on_gpu_memory("YUV420", 1920, 1080, device_handle);
```

#### Thread-Safe List (safe_list.h)

**Status**: Implementation incomplete

```cpp
class safe_list {
    ListNode* list_head;
    ListNode* list_tail;
    int size;
public:
    void clear();
};
```

## Dependencies

### Inter-module Dependencies

```
core/
├── config (Independent)
│   └── Depends on: nlohmann_json
├── operator (Depends on config)
│   └── Depends on: cuda, npp, FFmpeg, config
└── utils (Depends on config)
    └── Depends on: spdlog, FFmpeg, cuda, config
```

### External Dependencies

**config Module**:
- `nlohmann_json`: JSON parsing library

**operator Module**:
- `CUDA`: CUDA runtime
- `NPP`: NVIDIA Performance Primitives Library (nppicc, nppc, nppidei)
- `FFmpeg`: avcodec, avformat, avutil, swscale

**utils Module**:
- `spdlog`: Logging library
- `FFmpeg`: avcodec, avformat, avutil, swscale
- `CUDA`: CUDA runtime

## Build Configuration

### Top-level CMakeLists.txt

```cmake
set(CORE_TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(config)
add_subdirectory(operator)
add_subdirectory(utils)
```

### Module CMakeLists.txt

**config**:
```cmake
add_library(config STATIC src/config.cpp)
find_package(nlohmann_json REQUIRED)
target_link_libraries(config PRIVATE nlohmann_json::nlohmann_json)
```

**operator**:
```cmake
enable_language(CUDA)
add_library(operator_nvidia STATIC ...)
target_link_libraries(operator_nvidia cuda nppicc nppc nppidei config ...)
```

**utils**:
```cmake
add_library(utils STATIC src/tools.cpp)
target_link_libraries(utils PRIVATE avcodec avformat avutil swscale ...)
```

## Usage Flow

### 1. Initialize Configuration

```cpp
#include "config.h"

// Set configuration file at program startup
config::SetConfigFileName("config.json");

// Get configuration instance
auto& cfg = config::GetInstance();
```

### 2. Use Logging

```cpp
#include "log.hpp"

LOG_INFO("Application started");
LOG_DEBUG("Debug value: {}", value);
```

### 3. Use Queue

```cpp
#include "safe_queue.hpp"

safe_queue<Frame> queue;
// ... use queue ...
```

### 4. Use CUDA Kernels

```cpp
#include "nvidia_kernel.h"

// Call stitching kernel
launch_stitch_kernel_with_mapping_table(...);
```

## Performance Optimization

### 1. CUDA Kernel Optimization

- **Parallel Processing**: Y and UV planes processed in parallel using different CUDA streams
- **Texture Memory**: Mapping tables use CUDA texture memory to accelerate lookups
- **Memory Coalescing**: Optimize memory access patterns
- **Stream Parallelism**: Use multiple CUDA streams to improve parallelism

### 2. Queue Optimization

- **Lock-free Design**: Use lock-free data structures where possible
- **Batch Processing**: Process multiple elements in batches
- **Overflow Protection**: Automatically discard oldest elements to avoid memory overflow

### 3. Configuration Optimization

- **Singleton Pattern**: Avoid duplicate configuration loading
- **Lazy Loading**: Load configuration items on demand
- **Caching Mechanism**: Cache frequently used configuration items

## Common Issues

### 1. Configuration File Loading Failure

**Possible Causes**:
- Configuration file path error
- JSON format error
- File permission issues

**Solutions**:
- Check configuration file path
- Verify JSON format
- Check file permissions

### 2. CUDA Kernel Execution Failure

**Possible Causes**:
- CUDA device unavailable
- Insufficient memory
- Kernel parameter errors

**Solutions**:
- Check CUDA device status
- Check GPU memory
- Verify kernel parameters
- Use `CHECK_CUDA` macro to check errors

### 3. Queue Blocking

**Possible Causes**:
- Producer stopped but `stop()` not called
- Queue empty and no new data

**Solutions**:
- Ensure `stop()` is called when stopping
- Check if producer is running normally

### 4. Logging Not Output

**Possible Causes**:
- Log level set too high
- Logging system not initialized

**Solutions**:
- Check `loglevel` in configuration file
- Ensure configuration file is loaded

## Extension Development

### 1. Adding New Configuration Items

1. Add new configuration structure in `config.h`
2. Add loading logic in `config.cpp`
3. Update JSON configuration file format

### 2. Adding New CUDA Kernels

1. Create `.cuh` header file to define interface
2. Create `.cu` source file to implement kernel
3. Add source file in `CMakeLists.txt`
4. Include header file in `nvidia_kernel.h`

### 3. Adding New Utility Functions

1. Declare function in `tools.hpp`
2. Implement function in `tools.cpp`
3. Add necessary dependencies
