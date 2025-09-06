add_library(camera_manager STATIC
  src/camera_manager.cpp
  src/TaskManager.cpp
  src/Producer.cpp
  src/Consumer.cpp
  src/AVFrameProducer.cpp
  src/RtspConsumer.cpp
  src/StitchConsumer.cpp
  src/LogConsumer.cpp
)

target_include_directories(camera_manager PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/utils/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/rtsp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/data_processing/include
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
  data_processing
  config
)