set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(HEADERS 
    include/widget.h
    include/nv12render.h
    include/mainwindow.h
)

set(SOURCES
    src/widget.cpp
    src/nv12render.cpp
    src/mainwindow.cpp
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
  ${CMAKE_CURRENT_SOURCE_DIR}/../../core/rtsp/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../core/config/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../core/data_processing/include
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
  data_processing
  config
  camera_manager
)
