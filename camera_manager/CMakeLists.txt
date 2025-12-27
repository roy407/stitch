add_library(camera_manager STATIC
  src/camera_manager.cpp
  src/TaskManager.cpp
  src/Producer.cpp
  src/Consumer.cpp
  src/PacketProducer.cpp
  src/RTSPPacketProducer.cpp
  src/MP4PacketProducer.cpp
  src/USBPacketProducer.cpp
  src/SingleViewConsumer.cpp
  src/RtspConsumer.cpp
  src/StitchConsumer.cpp
  src/LogConsumer.cpp
  src/cuda_handle_init.cpp
  src/JetsonDecoderConsumer.cpp
  src/DecoderConsumer.cpp
  src/EncoderConsumer.cpp
  src/CallbackConsumer.cpp
  src/Pipeline.cpp
)

# 查找CUDA工具包
find_package(CUDAToolkit REQUIRED)

# 查找NVML库（现在可以通过CUDAToolkit找到）
if(TARGET CUDA::nvml)
    message(STATUS "Found NVML via CUDAToolkit")
    add_compile_definitions(ENABLE_NVML)
else()
    # 回退到原来的find_library
    find_library(NVML_LIB nvidia-ml)
    if(NVML_LIB)
        message(STATUS "Found NVML library: ${NVML_LIB}")
        add_compile_definitions(ENABLE_NVML)
    else()
        message(WARNING "NVML library not found")
    endif()
endif()

target_include_directories(camera_manager PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/utils/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/rtsp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/operator/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/operator/include/nvidia
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/config/include
    ${FFMPEG_INCLUDE_DIRS}
    # 添加NVML头文件路径
    /usr/local/cuda-11.8/targets/x86_64-linux/include
)

# 链接库文件
target_link_libraries(camera_manager PRIVATE 
    avcodec
    avformat
    avutil
    avdevice
    swscale
    pthread
    config
    operator_nvidia
    utils
    dl
    
    # CUDA相关库（使用CUDAToolkit提供的目标）
    CUDA::cudart      # CUDA运行时库
    CUDA::nvml        # NVML库（如果找到的话）
)

# 可选：如果需要添加CUDA编译选项
if(CMAKE_CUDA_COMPILER)
    # 设置CUDA编译选项
    set_property(TARGET camera_manager PROPERTY CUDA_ARCHITECTURES "75;80;86")
endif()
