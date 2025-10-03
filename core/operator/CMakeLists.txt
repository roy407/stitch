enable_language(CUDA)

add_library(operator_nvidia STATIC
  src/nvidia/scale.cu
  src/nvidia/stitch.cu
)

target_include_directories(operator_nvidia PUBLIC
  ${CORE_TOP_DIR}/operator/include/nvidia
  ${FFMPEG_INCLUDE_DIRS}
)

target_link_libraries(operator_nvidia
    avcodec avformat avutil swscale
    cuda nppicc nppc nppidei config 
)
