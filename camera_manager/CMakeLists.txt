add_library(camera_manager STATIC
  src/camera_manager.cpp
)

target_include_directories(camera_manager
  PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/safe_tools
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/core/rtsp/include
    ${CMAKE_SOURCE_DIR}/core/stitch/include
    ${CMAKE_SOURCE_DIR}/core/config/include
    ${FFMPEG_INCLUDE_DIR}
)

target_link_libraries(camera_manager PRIVATE 
avcodec
avformat
avutil
swscale
pthread
rtsp
stitch
config
# ${OPENCV_LIBS}
)