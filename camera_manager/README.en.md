# Camera Manager Module Documentation

## Overview

`camera_manager` is a high-performance multi-camera video processing framework that supports capture, decoding, stitching, and output of multiple video streams. This module adopts a producer-consumer pattern, manages multiple camera processing pipelines through Pipeline, supports various input sources such as RTSP, MP4 files, and USB cameras, and provides CUDA-based hardware-accelerated decoding and image stitching capabilities.

## Core Features

- **Multiple Input Sources**: RTSP streams, MP4 files, USB cameras
- **Hardware Acceleration**: CUDA-based hardware decoding and image processing
- **Multi-Camera Stitching**: Real-time stitching of multiple camera views
- **Multi-threaded Architecture**: TaskManager-based multi-threaded task management
- **Flexible Configuration**: Manage camera and pipeline parameters through JSON configuration files
- **Callback Mechanism**: Support frame data callbacks through CallbackConsumer

## Directory Structure

```
camera_manager/
├── include/          # Header files directory
│   ├── camera_manager.h      # Main manager class
│   ├── Pipeline.h            # Pipeline class
│   ├── Producer.h            # Producer base class
│   ├── PacketProducer.h      # Packet producer base class
│   ├── RTSPPacketProducer.h  # RTSP stream producer
│   ├── MP4PacketProducer.h   # MP4 file producer
│   ├── USBPacketProducer.h   # USB camera producer
│   ├── Consumer.h            # Consumer base class
│   ├── DecoderConsumer.h      # Decoder consumer
│   ├── JetsonDecoderConsumer.h # Jetson-specific decoder
│   ├── StitchConsumer.h      # Stitching consumer
│   ├── SingleViewConsumer.h  # Single view resize consumer
│   ├── RtspConsumer.h        # RTSP streaming consumer
│   ├── LogConsumer.h         # Logging statistics consumer
│   ├── CallbackConsumer.h    # Callback consumer
│   ├── Channel.h            # Channel classes (PacketChannel, FrameChannel)
│   ├── TaskManager.h         # Task manager base class
│   ├── IStitch.h            # Stitching interface
│   ├── StitchImpl.h         # Stitching implementation template class
│   └── cuda_handle_init.h   # CUDA device initialization
└── src/              # Source files directory
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

## Architecture Design

### 1. Core Class Hierarchy

```
TaskManager (Base Class)
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

### 2. Data Flow Architecture

```
[Input Source] → [PacketProducer] → [PacketChannel]
                                        ↓
                                    [DecoderConsumer] → [FrameChannel]
                                                                ↓
                                                        [StitchConsumer] → [FrameChannel] (Stitched Image)
                                                                                ↓
                                                                        [CallbackConsumer] (Callback Processing)
```

### 3. Pipeline Structure

Each `Pipeline` manages a complete processing flow:

1. **Multiple PacketProducers**: One producer per camera
2. **Multiple DecoderConsumers**: One decoder per camera
3. **Optional SingleViewConsumer**: Single view resizing (if enabled)
4. **One StitchConsumer**: Multi-camera stitching
5. **Optional RtspConsumer**: RTSP streaming output
6. **Optional CallbackConsumer**: Frame data callback

## Core Components

### 1. camera_manager (Main Manager)

**Responsibilities**:
- Manage all Pipeline instances
- Provide singleton access interface
- Unified start/stop for all pipelines
- Provide stream access interface

**Key Interfaces**:
```cpp
static camera_manager* GetInstance();  // Get singleton
void start();                           // Start all pipelines
void stop();                            // Stop all pipelines
void initPipeline();                    // Initialize pipelines
FrameChannel* getStitchCameraStream(int pipeline_id) const;  // Get stitched stream
FrameChannel* getSingleCameraSubStream(int cam_id) const;     // Get single camera stream
size_t getResizeCameraStreamCount() const;                    // Get resize stream count
```

**Usage Example**:
```cpp
auto* mgr = camera_manager::GetInstance();
mgr->start();
FrameChannel* stitch_stream = mgr->getStitchCameraStream(0);
```

### 2. Pipeline

**Responsibilities**:
- Create and manage all Producers and Consumers based on configuration
- Connect Channels between components
- Manage the lifecycle of the entire processing flow

**Key Interfaces**:
```cpp
Pipeline(int pipeline_id);                    // Construct from pipeline_id
Pipeline(const PipelineConfig& p);             // Construct from config
static void setLogConsumer(LogConsumer* log);  // Set log consumer
void start();                                  // Start pipeline
void stop();                                   // Stop pipeline
FrameChannel* getStitchCameraStream() const;   // Get stitched stream
FrameChannel* getResizeCameraStream(int cam_id) const;  // Get resize stream
size_t getResizeCameraStreamCount() const;      // Get resize stream count
```

**Initialization Flow**:
1. Create `PacketProducer` for each camera (based on config type: RTSP/MP4/USB)
2. If RTSP streaming is enabled, create `RtspConsumer` and connect to Producer
3. Create `DecoderConsumer` for each camera (normal decoder or Jetson decoder)
4. If single view is enabled, create `SingleViewConsumer`
5. Create `StitchConsumer` for multi-camera stitching
6. All components connected through Channels

### 3. TaskManager

**Responsibilities**:
- Provide thread management functionality
- Unified task start/stop interface
- Thread lifecycle management

**Key Interfaces**:
```cpp
TaskManager();           // Constructor
virtual ~TaskManager();  // Destructor
virtual void start();    // Start thread
virtual void stop();     // Stop thread
virtual void run() = 0;  // Pure virtual function, implemented by subclasses
```

**Implementation Mechanism**:
- Use `std::thread` to manage threads
- `start()` creates thread and calls `run()`
- `stop()` sets `running=false` and waits for thread to end
- Each thread has a unique name identifier

### 4. PacketProducer

**Responsibilities**:
- Read video packets (AVPacket) from input sources
- Send packets to `PacketChannel`
- Support multiple input source types

**Subclass Implementations**:

#### RTSPPacketProducer
- Read data from RTSP streams
- Support automatic reconnection mechanism
- Configuration options: `buffer_size`, `rtsp_transport`, `stimeout`
- Support TCP transport mode

#### MP4PacketProducer
- Read data from MP4 files
- Support file loop playback

#### USBPacketProducer
- Read data from USB cameras (V4L2)
- Use FFmpeg's `v4l2` input format

**Data Flow**:
```
Input Source → AVPacket → PacketChannel (2 channels)
                      ├── m_channel2rtsp (RTSP streaming)
                      └── m_channel2decoder (Decoding)
```

**Key Interfaces**:
```cpp
int getWidth() const;                    // Get width
int getHeight() const;                   // Get height
AVRational getTimeBase() const;          // Get time base
AVCodecParameters* getAVCodecParameters() const;  // Get codec parameters
PacketChannel* getChannel2Rtsp() const;  // Get RTSP channel
PacketChannel* getChannel2Decoder() const;  // Get decoder channel
```

### 5. DecoderConsumer

**Responsibilities**:
- Receive AVPacket from `PacketChannel`
- Decode to AVFrame using hardware decoder (CUDA)
- Send decoded frames to `FrameChannel`

**Supported Decoders**:
- `h264_cuvid`: NVIDIA CUDA hardware H.264 decoder
- `hevc_cuvid`: NVIDIA CUDA hardware HEVC decoder
- Other FFmpeg-supported hardware decoders

**Data Flow**:
```
PacketChannel → AVPacket → avcodec_send_packet()
                          → avcodec_receive_frame()
                          → AVFrame (CUDA) → FrameChannel (2 channels)
                                              ├── m_channel2stitch (Stitching)
                                              └── m_channel2resize (Resizing)
```

**Key Implementation**:
- Use `cuda_handle_init` to get CUDA device handle
- Decoded frame format is `AV_PIX_FMT_CUDA`
- Create frame references (`av_frame_ref`) for stitching and resizing separately
- Record decoding timestamps

**Key Interfaces**:
```cpp
DecoderConsumer(const std::string& codec_name);  // Constructor
void setAVCodecParameters(AVCodecParameters* codecpar, AVRational time_base);  // Set codec parameters
void setChannel(PacketChannel* channel);         // Set input channel
FrameChannel* getChannel2Resize();               // Get resize channel
FrameChannel* getChannel2Stitch();               // Get stitch channel
```

### 6. StitchConsumer

**Responsibilities**:
- Receive decoded frames from multiple `FrameChannel`s
- Call CUDA stitching kernel for multi-camera stitching
- Send stitching results to output `FrameChannel`
- Support multiple output channels (display and RTSP streaming)

**Key Features**:
- Support multiple stitching modes: `mapping_table`, `raw`
- Support multiple pixel formats: `YUV420`, `YUV420P`
- Call templated stitching implementation through `StitchOps` interface
- Support KERNEL_TEST mode for kernel testing

**Data Flow**:
```
Multiple FrameChannels → Collect all camera frames
                → StitchOps::stitch() (CUDA stitching)
                → AVFrame (Stitching result)
                → FrameChannel (2 outputs)
                    ├── m_channel2show (Display)
                    └── m_channel2rtsp (RTSP streaming)
```

**Key Interfaces**:
```cpp
StitchConsumer(StitchOps* ops, int single_width, int height, int output_width);  // Constructor
void setChannels(std::vector<FrameChannel*> channels);  // Set input channels
FrameChannel* getChannel2Show();   // Get display channel
FrameChannel* getChannel2Rtsp();   // Get RTSP channel
```

**Implementation Details**:
- Use `av_frame_ref` to create frame references, avoiding data copying
- Record stitching timestamps for performance analysis
- Support resource cleanup on graceful exit

### 7. SingleViewConsumer

**Responsibilities**:
- Receive decoded frames from a single camera
- Use CUDA for image resizing (resize functionality currently commented out)
- Send processed frames to output `FrameChannel`

**Use Cases**:
- Single camera preview
- Low resolution display

**Key Interfaces**:
```cpp
SingleViewConsumer(int width, int height, float scale_factor);  // Scale by factor
SingleViewConsumer(int width, int height, AVRational rational);  // Scale by ratio
SingleViewConsumer(int width, int height, int output_width, int output_height);  // Specify output size
void setChannel(FrameChannel* channel);  // Set input channel
FrameChannel* getChannel2Show() const;   // Get output channel
```

### 8. CallbackConsumer

**Responsibilities**:
- Receive frame data from `FrameChannel`
- Pass frame data to external processing through callback function
- Support timestamp recording and performance analysis

**Design Features**:
- Provide standardized callback interface
- Encapsulate thread-safe frame data transmission
- Built-in timestamp recording mechanism
- Internally manage independent consumer thread

**Usage Example**:
```cpp
auto consumer = std::make_shared<CallbackConsumer>();
consumer->setChannel(frameChannel);
consumer->setCallback([](Frame frame) {
    // Process received frame data
    processFrame(frame);
});
consumer->setPipelineName("pipeline_0");
consumer->setTimingWatcher(true);  // Enable timing monitoring
consumer->start();
```

**Key Interfaces**:
```cpp
void setChannel(FrameChannel* channel);              // Set input channel
void setCallback(Callback_Handle callback);          // Set callback function
void setPipelineName(std::string name);             // Set pipeline name
void setTimingWatcher(bool enable);                 // Enable/disable timing monitoring
```

### 9. Channel

**Responsibilities**:
- Provide thread-safe data transmission channel
- Producer-consumer pattern based on queue implementation

**Types**:

#### PacketChannel
```cpp
class PacketChannel {
    list_queue<Packet> m_data;  // or safe_queue<Packet>
public:
    bool recv(Packet& out);  // Blocking receive
    void send(Packet& p);    // Send
    void clear();            // Clear
    void stop();             // Stop
};
```

#### FrameChannel
```cpp
class FrameChannel {
    list_queue<Frame> m_data;  // or safe_queue<Frame>
public:
    bool recv(Frame& out);   // Blocking receive
    void send(Frame& p);     // Send
    void clear();            // Clear
    void stop();             // Stop
};
```

**Features**:
- Use `wait_and_pop` for blocking receive
- Thread-safe data transmission
- Support stop signal
- Support queue clearing

### 10. StitchImpl

**Responsibilities**:
- Provide templated stitching implementation
- Manage CUDA memory and kernel parameters
- Execute actual stitching operations

**Template Parameters**:
- `Format`: Pixel format (`YUV420`, `YUV420P`)
- `KernelTag`: Stitching kernel type (`MappingTableKernel`, `RawKernel`, `HMatrixInvKernel`, etc.)

**Supported Stitching Kernels**:
- `MappingTableKernel`: Mapping table-based stitching (most common)
- `RawKernel`: Raw stitching
- `HMatrixInvKernel`: Inverse homography matrix-based stitching
- `HMatrixInvV1_1Kernel`: Version 1.1 inverse homography matrix stitching
- `HMatrixInvV2Kernel`: Version 2 inverse homography matrix stitching
- `CropKernel`: Crop kernel (untested)

**Implementation Flow**:
1. `init_impl()`: Allocate CUDA memory, initialize kernel parameters
2. `do_stitch_impl()`:
   - Copy input frame data to GPU
   - Allocate output frame buffer
   - Call CUDA kernel to execute stitching
   - Return stitching result

**StitchOps Interface**:
```cpp
struct StitchOps {
    using StitchFunc = AVFrame* (*)(void*, AVFrame**);
    using InitFunc   = void (*)(void*, int, int, int, int);
    
    void* obj = nullptr;
    StitchFunc stitch = nullptr;
    InitFunc init = nullptr;
};

// Create StitchOps
template<typename Impl>
StitchOps* make_stitch_ops(Impl* obj);

// Delete StitchOps
template<typename Impl>
void delete_stitch_ops(StitchOps* ops);
```

### 11. LogConsumer

**Responsibilities**:
- Periodically print statistics of Producers and Consumers
- Monitor performance metrics such as frame rate and timestamps
- Print GPU/CPU status

**Statistics**:
- Frame count
- Timestamp
- Frame rate calculation
- GPU usage (if supported)

**Key Interfaces**:
```cpp
void setProducer(PacketProducer* pro);   // Register producer
void setConsumer(StitchConsumer* con);   // Register consumer
```

### 12. RtspConsumer

**Responsibilities**:
- Receive packets from `PacketChannel`
- Push packets to RTSP server
- Manage RTSP output stream

**Key Interfaces**:
```cpp
RtspConsumer(const std::string& push_stream_url);  // Constructor
void setChannel(PacketChannel* m_channel);         // Set input channel
void setParamters(AVCodecParameters* codecpar, AVRational time_base);  // Set parameters
```

### 13. JetsonDecoderConsumer

**Responsibilities**:
- Jetson platform-specific decoder implementation
- Optimize Jetson hardware decoding performance

**Differences from DecoderConsumer**:
- Use Jetson-specific decoder implementation
- May use different hardware acceleration paths

## Configuration System

### Configuration File Structure

Access configuration through `CFG_HANDLE`, including:

**PipelineConfig**:
- `pipeline_id`: Pipeline ID
- `enable`: Whether enabled
- `default_width`, `default_height`: Default resolution
- `cameras`: Camera configuration list
- `stitch`: Stitching configuration
  - `stitch_mode`: Stitching mode (`"mapping_table"`, `"raw"`)
  - `stitch_impl`: Stitching implementation configuration

**CameraConfig**:
- `cam_id`: Camera ID
- `name`: Camera name
- `input_url`: Input URL (RTSP/file path/USB device)
- `width`, `height`: Resolution
- `enable_view`: Whether to enable single view
- `scale_factor`: Scale factor
- `rtsp`: Whether to enable RTSP streaming
- `output_url`: RTSP streaming address

**GlobalConfig**:
- `type`: Input type (`"rtsp"`, `"mp4"`, `"usb"`)
- `format`: Pixel format (`"YUV420"`, `"YUV420P"`)
- `decoder`: Decoder name (`"h264_cuvid"`, `"jetson"`, etc.)
- `encoder`: Encoder name

### Configuration Example

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

## Usage Flow

### 1. Initialization

```cpp
#include "camera_manager.h"

// Get singleton (automatically initializes all Pipelines)
auto* mgr = camera_manager::GetInstance();
```

### 2. Start

```cpp
mgr->start();  // Start all Pipelines
```

### 3. Get Streams

```cpp
// Get stitched stream
FrameChannel* stitch_stream = mgr->getStitchCameraStream(0);

// Get single camera stream
FrameChannel* single_stream = mgr->getSingleCameraSubStream(0);
```

### 4. Receive Frames

```cpp
Frame frame;
while (stitch_stream->recv(frame)) {
    AVFrame* avframe = frame.m_data;
    // Process frame data...
    av_frame_free(&frame.m_data);
}
```

### 5. Use Callback Consumer

```cpp
#include "CallbackConsumer.h"

auto callback_consumer = std::make_shared<CallbackConsumer>();
callback_consumer->setChannel(stitch_stream);
callback_consumer->setCallback([](Frame frame) {
    // Process frame data
    processFrame(frame.m_data);
    av_frame_free(&frame.m_data);
});
callback_consumer->setPipelineName("pipeline_0");
callback_consumer->start();
```

### 6. Stop

```cpp
mgr->stop();  // Stop all Pipelines
```

## Dependencies

### System Libraries
- `pthread`: Thread support

### FFmpeg Libraries
- `avcodec`: Codec
- `avformat`: Format handling
- `avutil`: Utility functions
- `avdevice`: Device support (V4L2)
- `swscale`: Image scaling (partial functionality)

### CUDA Libraries
- `cuda`: CUDA runtime

### Internal Project Libraries
- `config`: Configuration management
- `operator_nvidia`: NVIDIA operators (stitching kernels)
- `utils`: Utility functions (includes list_queue or safe_queue)

### Third-party Libraries
- `opencv`: Image processing (partial functionality, optional)
- `onnxruntime`: ONNX inference (optional)

## Build Configuration

### CMakeLists.txt Key Configuration

```cmake
# Include directories
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

# Link libraries
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

## Performance Optimization

### 1. Hardware Acceleration
- Use CUDA hardware decoders (`h264_cuvid`, `hevc_cuvid`)
- All image processing done on GPU
- Avoid CPU-GPU data transfer

### 2. Multi-threaded Architecture
- Each Producer/Consumer runs in independent thread
- Thread communication through Channels
- Avoid lock contention, use lock-free queues

### 3. Memory Management
- Use FFmpeg's frame pool (`hw_frames_ctx`)
- Reuse GPU memory buffers
- Release unused frames promptly
- Use `av_frame_ref` to avoid unnecessary copying

### 4. Frame Reference Mechanism
- Use `av_frame_ref` in StitchConsumer to create frame references
- Multiple output channels share the same frame data
- Reduce memory copy overhead

## Common Issues

### 1. RTSP Connection Failure
- Check network connection
- Verify RTSP URL is correct
- Check firewall settings
- Automatic reconnection mechanism already implemented in code

### 2. Decoding Failure
- Confirm using hardware decoder (`h264_cuvid`)
- Check if CUDA device is available
- Verify input stream format is supported
- Check CUDA device handle initialization

### 3. Stitching Failure
- Check stitching configuration (mapping table, output size)
- Confirm all camera frames have been received
- Check if CUDA memory is sufficient
- Verify stitching kernel parameters are correct

### 4. Frame Loss
- Check Channel buffer size
- Confirm Consumer processing speed is fast enough
- Monitor frame count statistics
- Check if threads exit normally

### 5. Memory Leaks
- Ensure all `av_frame_alloc()` have corresponding `av_frame_free()`
- Check reference count after `av_frame_ref()`
- Use `av_frame_unref()` to decrease reference count
- Clean up all resources on exit

### 6. Thread Exit Issues
- Ensure `stop()` is called correctly
- Check if `running` flag is set correctly
- Confirm Channel's `stop()` is called
- Wait for all threads to exit normally

## Extension Development

### Adding New Input Source Types

1. Inherit from `PacketProducer`
2. Implement `run()` method to read data
3. Add creation logic in `Pipeline.cpp`:

```cpp
if(type == "new_type") {
    pro = new NewTypePacketProducer(cam);
}
```

### Adding New Stitching Kernels

1. Define new `KernelTag` struct in `IStitch.h`:

```cpp
struct NewKernel {
    // Kernel parameters
    bool initgetKernelGpuMemory(int num);
    void freeKernelGpuMemory();
};
```

2. Implement `initgetKernelGpuMemory()` and `freeKernelGpuMemory()`
3. Add support in `Pipeline.cpp`'s `getStitchConsumer()`

### Adding New Consumer Types

1. Inherit from `Consumer`
2. Implement `run()` method to process frames
3. Add creation and connection logic in `Pipeline.cpp`

### Using CallbackConsumer for Custom Processing

```cpp
auto consumer = std::make_shared<CallbackConsumer>();
consumer->setChannel(frameChannel);
consumer->setCallback([](Frame frame) {
    // Custom processing logic
    customProcess(frame);
});
consumer->start();
```

## Debugging Tips

### 1. Enable Logging
- Use `LogConsumer` to monitor performance
- Check frame rate and timestamps
- Monitor GPU usage

### 2. Timestamp Analysis
- Use `CallbackConsumer`'s `setTimingWatcher(true)` to enable timing monitoring
- Analyze processing time at each stage
- Locate performance bottlenecks

### 3. KERNEL_TEST Mode
- Define `KERNEL_TEST` at compile time
- Used for testing stitching kernel performance
- Avoid actual camera input

### 4. Resource Monitoring
- Monitor CUDA memory usage
- Check thread count
- Monitor Channel queue length
