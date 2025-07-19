add_library(stitch STATIC
  src/cann_handle.cpp
  src/image_decoder.cpp
  # src/image_encoder.cpp
  # src/Stitch.cpp
  # src/stitch.cu
  # src/scale.cu
)

link_directories(${FFMPEG_LIB_DIR})

target_include_directories(stitch
  PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../safe_tools
  ${CMAKE_SOURCE_DIR}/core/config/include
  ${FFMPEG_INCLUDE_DIR}
  # /usr/local/include/opencv4
)

target_link_libraries(stitch
    avcodec avformat avutil swscale
    config 
    ascendcl        # 基础运行库
    acl_dvpp        # 图像处理库
    acl_op_compiler # 算子编译器
)
