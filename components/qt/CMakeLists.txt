# /home/eric/stitch/components/qt/CMakeLists.txt
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(HEADERS 
    include/visible_camera_widget.h
    include/infrared_camera_widget.h
    include/nv12render.h
    include/mainwindow.h
    include/visible_single_camera_widget.h
    include/widget_for_test.h
    include/full_stitch_widget.h
)

set(SOURCES
    src/visible_camera_widget.cpp
    src/infrared_camera_widget.cpp
    src/nv12render.cpp
    src/mainwindow.cpp
    src/visible_single_camera_widget.cpp
    src/widget_for_test.cpp
    src/full_stitch_widget.cpp
)

add_library(qt STATIC
    ${HEADERS}
    ${SOURCES}
)
find_package(OpenGL REQUIRED)

target_include_directories(qt PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../camera_manager/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../core/utils/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../core/shm/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../core/rtsp/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../core/config/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../core/operator/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../core/operator/include/nvidia
  ${FFMPEG_INCLUDE_DIRS}
)
target_link_libraries(qt PRIVATE Qt5::Widgets)
target_link_options(qt PRIVATE -Wl,-rpath-link=/usr/local/lib)

target_link_libraries(qt PRIVATE
  Qt5::Core
  Qt5::Widgets
  OpenGL::GL
  avcodec
  avformat
  avutil
  swscale
  pthread
  config
  camera_manager
  utils
)
