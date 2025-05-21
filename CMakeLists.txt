cmake_minimum_required(VERSION 3.10)
project(pstitch CUDA CXX)

set(CMAKE_CXX_STANDARD 14)

# 收集所有动态库文件
file(GLOB OPENCV_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/cv2/lib/*.so")
file(GLOB FFMPEG_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg/lib/*.so")

# 添加可执行文件
add_executable(my_stitch_app main.cpp)

# 添加子目录
add_subdirectory(rtsp)
add_subdirectory(stitch)

target_include_directories(my_stitch_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cv2/include/opencv4)
target_include_directories(my_stitch_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg/include)

# 链接库文件
target_link_libraries(my_stitch_app PRIVATE 
${OPENCV_LIBS}
${FFMPEG_LIBS}
pthread
rtsp
stitch
)