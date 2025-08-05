cmake_minimum_required(VERSION 3.10)
project(pstitch CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 17)

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

if(BUILD_SHARED_LIB)
    add_library(stitch_app SHARED lib.cpp)
else()
    add_executable(stitch_app main.cpp)
endif()

add_subdirectory(core)
add_subdirectory(camera_manager)

target_link_libraries(stitch_app PRIVATE 
    camera_manager
    ascendcl         # CANN 基础运行时库
    acl_op_compiler  # CANN 算子编译器库
    hccl             # 分布式通信库
    rt
    pthread
    dl
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