cmake_minimum_required(VERSION 3.10)
project(pstitch CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
<<<<<<< Updated upstream
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 用于检测野指针问题，但是打开后会导致cuInit创建失败，显示OUT_OF_MEMORY
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -g")
# 这个问题是 已知现象，原因不是你真的“内存不够”，而是 ASan 与 CUDA（尤其是 NVIDIA 驱动的 
# libcuda.so, libnvidia-fatbinaryloader.so, libnpp*, libnvrtc 等） 在运行时存在冲突。

option(BUILD_SHARED_LIB "Build as shared library instead of executable" OFF)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
find_package(Qt5 COMPONENTS Widgets REQUIRED)

=======

option(BUILD_SHARED_LIB "Build as shared library instead of executable" OFF)

set(CANN_PATH "/usr/local/Ascend/ascend-toolkit/latest")
set(ASCEND_ROOT ${CANN_PATH})
set(FFMPEG_INCLUDE_DIR "/root/Documents/FFmpeg-n4.4.4/ascend/include")
set(FFMPEG_LIB_DIR "/root/Documents/FFmpeg-n4.4.4/ascend/lib")

include_directories(
    ${ASCEND_ROOT}/include
    ${ASCEND_ROOT}/runtime/include
    ${FFMPEG_INCLUDE_DIR}
)

link_directories(
    ${ASCEND_ROOT}/lib64
    ${ASCEND_ROOT}/runtime/lib64
    ${FFMPEG_LIB_DIR}
)
>>>>>>> Stashed changes

if(BUILD_SHARED_LIB)
    add_library(stitch_app SHARED lib.cpp)
else()
    add_executable(stitch_app main.cpp)
endif()

add_subdirectory(core/rtsp)
add_subdirectory(core/stitch)
add_subdirectory(core/config)
add_subdirectory(camera_manager)
add_subdirectory(core/qt)

target_link_libraries(stitch_app PRIVATE 
<<<<<<< Updated upstream
camera_manager
Qt5::Widgets
qt
=======
    camera_manager
    ascendcl         # CANN 基础运行时库
    acl_op_compiler  # CANN 算子编译器库
    hccl             # 分布式通信库
    dvpp
    rt
    pthread
    dl
>>>>>>> Stashed changes
)

if(BUILD_SHARED_LIB)
    set_target_properties(rtsp PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(stitch PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(config PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(camera_manager PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

set(RESOURCE_DIR ${CMAKE_SOURCE_DIR}/resource)
set(DEST_DIR ${CMAKE_BINARY_DIR}/resource)

add_custom_command(
    OUTPUT ${DEST_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${RESOURCE_DIR} ${DEST_DIR}
    COMMENT "Copying resource directory..."
)

add_custom_target(copy_resources ALL DEPENDS ${DEST_DIR})