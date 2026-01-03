cmake_minimum_required(VERSION 3.10)
project(pstitch CUDA CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 用于检测野指针问题，但是打开后会导致cuInit创建失败，显示OUT_OF_MEMORY
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -g")
# 这个问题是 已知现象，原因不是你真的“内存不够”，而是 ASan 与 CUDA（尤其是 NVIDIA 驱动的 
# libcuda.so, libnvidia-fatbinaryloader.so, libnpp*, libnvrtc 等） 在运行时存在冲突。

option(BUILD_SHARED_LIB "Build as shared library instead of executable" OFF)
option(ENABLE_KERNEL_TEST "Enable kernel test features" OFF)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(spdlog REQUIRED)

# 设置ffmpeg的配置
set(FFMPEG_INCLUDE_DIRS "/usr/local/include")

if(BUILD_SHARED_LIB)
    add_library(stitch_app SHARED stitch_shared.cpp)
else()
    add_executable(stitch_app main.cpp)
endif()

add_subdirectory(core)
add_subdirectory(components)
add_subdirectory(camera_manager)

if(ENABLE_KERNEL_TEST)
    target_compile_definitions(camera_manager PRIVATE KERNEL_TEST)
endif()

target_link_directories(stitch_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/components/qt/include
)

target_link_libraries(stitch_app PRIVATE 
    camera_manager
    Qt5::Widgets
    qt
    spdlog::spdlog
)

if(BUILD_SHARED_LIB)
    set_target_properties(stitch PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(config PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(camera_manager PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

file(COPY ${CMAKE_SOURCE_DIR}/resource
     DESTINATION ${CMAKE_BINARY_DIR}
     FILES_MATCHING PATTERN "*")
file(COPY ${CMAKE_SOURCE_DIR}/scripts
     DESTINATION ${CMAKE_BINARY_DIR}
     FILES_MATCHING PATTERN "*")