enable_language(CUDA)

add_library(stitch STATIC
  src/cuda_handle_init.cpp
  src/image_decoder.cpp
  src/image_encoder.cpp
  src/Stitch.cpp
  src/stitch.cu
  src/scale.cu
)

link_directories(/usr/local/lib)

target_include_directories(stitch
  PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../safe_tools
  ${CMAKE_SOURCE_DIR}/core/config/include
  /usr/local/include
  # /usr/local/include/opencv4
)

target_link_libraries(stitch
    avcodec avformat avutil swscale
    cuda nppicc nppc nppidei config 
    # ${OPENCV_LIBS} opencv_core opencv_imgproc opencv_cudawarping
)
