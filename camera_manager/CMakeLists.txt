add_library(camera_manager STATIC
  src/camera_manager.cpp
  src/TaskManager.cpp
  src/Producer.cpp
  src/Consumer.cpp
  src/AVFrameProducer.cpp
  src/ResizeConsumer.cpp
  src/RtspConsumer.cpp
  src/StitchConsumer.cpp
  src/LogConsumer.cpp
  src/cuda_handle_init.cpp
  src/image_decoder.cpp
  src/image_encoder.cpp
)

target_include_directories(camera_manager PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/utils/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/rtsp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/operator/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/operator/include/nvidia
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/config/include
    ${FFMPEG_INCLUDE_DIRS}
)

# 链接库文件
target_link_libraries(camera_manager PRIVATE 
  avcodec
  avformat
  avutil
  swscale
  pthread
  config
  cuda 
  operator_nvidia
)