
enable_language(CUDA)

add_library(stitch STATIC
  src/cuda_handle_init.cpp
  src/image_encoder.cpp
  src/Stitch.cpp
  src/stitch.cu
)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg/lib)

target_include_directories(stitch
  PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/safe_tools
  ${CMAKE_SOURCE_DIR}/ffmpeg/include
)

target_link_libraries(stitch
    avcodec avformat avutil swscale
    cuda nppicc nppc nppidei
)
