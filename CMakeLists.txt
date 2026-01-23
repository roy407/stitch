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

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(spdlog REQUIRED)

# 设置ffmpeg的配置
set(FFMPEG_INCLUDE_DIRS "/usr/local/include")

if(BUILD_SHARED_LIB)
    add_library(stitch_app SHARED lib.cpp)
else()
    add_executable(stitch_app main.cpp)
endif()

add_subdirectory(core)
add_subdirectory(components)
add_subdirectory(camera_manager)

target_link_directories(stitch_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/components/qt/include
)

# 查找 OpenCV（用于检测功能）
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(OpenCV opencv4)
    if(NOT OpenCV_FOUND)
        pkg_check_modules(OpenCV opencv)
    endif()
endif()
if(NOT OpenCV_FOUND)
    find_package(OpenCV QUIET)
endif()

if(OpenCV_FOUND OR OpenCV_INCLUDE_DIRS)
    if(OpenCV_INCLUDE_DIRS)
        target_include_directories(stitch_app PRIVATE ${OpenCV_INCLUDE_DIRS})
    endif()
endif()

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

set(RESOURCE_DIR ${CMAKE_SOURCE_DIR}/resource)
set(DEST_DIR ${CMAKE_BINARY_DIR}/resource)

add_custom_command(
    OUTPUT ${DEST_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${RESOURCE_DIR} ${DEST_DIR}
    COMMENT "Copying resource directory..."
)

add_custom_target(copy_resources ALL DEPENDS ${DEST_DIR})