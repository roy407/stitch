
enable_language(CUDA)

add_library(stitch STATIC
  src/image_decoder.cpp
  src/image_encoder.cpp
  src/Stitch.cpp
  src/stitch.cu
)

target_include_directories(stitch
  PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(stitch
    avcodec avformat avutil swscale
    cuda nppicc nppc nppidei
)
