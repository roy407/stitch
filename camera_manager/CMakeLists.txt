# /home/eric/stitch/camera_manager/CMakeLists.txt
add_library(camera_manager SHARED
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

#隐藏所有接口，只能用API暴露
# set(CMAKE_C_VISIBILITY_PRESET hidden)
# set(CMAKE_CXX_VISIBILITY_PRESET hidden)
# set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# 确保库输出到lib目录
set_target_properties(camera_manager PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
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
  avdevice
  swscale
  pthread
  config
  cuda 
  operator_nvidia
  utils
)