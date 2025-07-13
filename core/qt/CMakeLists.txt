add_library(qt STATIC
    src/mainwindow.cpp
    include/mainwindow.h  #用于触发moc编译
)

target_include_directories(qt
  PUBLIC

  ${CMAKE_SOURCE_DIR}/core/safe_tools
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/core/rtsp/include
  ${CMAKE_SOURCE_DIR}/core/stitch/include
  ${CMAKE_SOURCE_DIR}/core/config/include
  ${CMAKE_SOURCE_DIR}/camera_manager/include
  /usr/local/include
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
  stitch
  config
  camera_manager
)
