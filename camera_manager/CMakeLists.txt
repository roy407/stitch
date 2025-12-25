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

# 查找NVML库
find_library(NVML_LIB nvidia-ml)
if(NVML_LIB)
    message(STATUS "Found NVML library: ${NVML_LIB}")
    add_compile_definitions(ENABLE_NVML)
else()
    message(WARNING "NVML library not found")
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
  cuda 
  operator_nvidia
  utils
  ${NVML_LIB}  # 添加NVML库
  dl           # 添加动态链接库（如果使用dlopen）
)