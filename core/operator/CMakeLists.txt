enable_language(CUDA)

add_library(operator_nvidia STATIC
  src/nvidia/scale.cu
  src/nvidia/stitch_raw.cu
  src/nvidia/stitch_with_crop.cu
  src/nvidia/stitch_with_h_matrix_inv.cu
  src/nvidia/stitch_with_h_matrix.cu
  src/nvidia/stitch_with_mapping_table.cu
)

target_include_directories(operator_nvidia PUBLIC
  ${CORE_TOP_DIR}/operator/include/nvidia
  ${FFMPEG_INCLUDE_DIRS}
)

target_link_libraries(operator_nvidia
    avcodec avformat avutil swscale
    cuda nppicc nppc nppidei config 
)
