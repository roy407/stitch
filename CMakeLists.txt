cmake_minimum_required(VERSION 3.10)
project(pstitch CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
# 用于检测野指针问题，但是打开后会导致cuInit创建失败，显示OUT_OF_MEMORY
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -g")
# 这个问题是 已知现象，原因不是你真的“内存不够”，而是 ASan 与 CUDA（尤其是 NVIDIA 驱动的 
# libcuda.so, libnvidia-fatbinaryloader.so, libnpp*, libnvrtc 等） 在运行时存在冲突。

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(my_stitch_app main.cpp)

add_subdirectory(core/rtsp)
add_subdirectory(core/stitch)
add_subdirectory(core/config)
add_subdirectory(camera_manager)

target_link_libraries(my_stitch_app PRIVATE 
camera_manager
)