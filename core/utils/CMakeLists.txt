add_library(utils STATIC
    src/tools.cpp
)

target_include_directories(utils PRIVATE
    ${CORE_TOP_DIR}/utils/include
    ${CORE_TOP_DIR}/config/include
)

# 链接库文件
target_link_libraries(utils PRIVATE 
  avcodec
  avformat
  avutil
  swscale
  pthread
  config
  cuda
)
