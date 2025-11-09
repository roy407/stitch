add_library(data_processing STATIC
  src/cuda_handle_init.cpp
  src/image_decoder.cpp
  src/image_encoder.cpp
  src/Stitch.cpp
)

target_include_directories(data_processing PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CORE_TOP_DIR}/utils/include
  ${CORE_TOP_DIR}/config/include
  ${CORE_TOP_DIR}/operator/include/nvidia
  ${FFMPEG_INCLUDE_DIRS}
)

target_link_libraries(data_processing PRIVATE
    avcodec avformat avutil swscale
    cuda nppicc nppc nppidei config 
    operator_nvidia
)