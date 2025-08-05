add_library(data_processing STATIC
  src/cuda_handle_init.cpp
  src/image_decoder.cpp
  src/image_encoder.cpp
  src/Stitch.cpp
)

link_directories(${FFMPEG_LIB_DIR})

target_include_directories(data_processing
  PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/core/safe_tools
  ${CMAKE_SOURCE_DIR}/core/config/include
  ${CMAKE_SOURCE_DIR}/core/operator/include
  ${FFMPEG_INCLUDE_DIR}
)

target_link_libraries(data_processing
    avcodec avformat avutil swscale
    config
    operator
    ascendcl        # 基础运行库
    acl_dvpp        # 图像处理库
    acl_op_compiler # 算子编译器
)
