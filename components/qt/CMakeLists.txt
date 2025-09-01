add_library(qt STATIC
    src/mainwindow.cpp
    include/mainwindow.h  #用于触发moc编译
)

target_include_directories(qt PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../camera_manager/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../core/utils/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../core/rtsp/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../core/data_processing/include
  ${FFMPEG_INCLUDE_DIRS}
)
target_link_libraries(qt PRIVATE Qt5::Widgets)
target_link_options(qt PRIVATE -Wl,-rpath-link=/usr/local/lib)

target_link_libraries(qt PRIVATE
  Qt5::Widgets
  avcodec
  avformat
  avutil
  swscale
  pthread
  rtsp
  data_processing
  config
  camera_manager
)
