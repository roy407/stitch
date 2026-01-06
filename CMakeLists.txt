cmake_minimum_required(VERSION 3.10)
project(pstitch CUDA CXX C)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

option(ENABLE_KERNEL_TEST "Enable kernel test features" OFF)
option(BUILD_C_VERSION "Build C version executable" OFF)  # 新增：选择构建C版本

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(spdlog REQUIRED)

set(FFMPEG_INCLUDE_DIRS "/usr/local/include")

add_subdirectory(core)
add_subdirectory(components)
add_subdirectory(camera_manager)

# ========== 创建C接口库（仅当构建C版本时） ==========
if(BUILD_C_VERSION)
    message(STATUS "Building C version executable")
    
    # 创建C接口库
    add_library(stitch_c_interface STATIC
        src/cpp_exposure_to_c.cpp
    )
    
    # C接口库配置
    target_include_directories(stitch_c_interface PRIVATE
        ${CMAKE_SOURCE_DIR}/inc
        ${CMAKE_SOURCE_DIR}/camera_manager/include
        ${FFMPEG_INCLUDE_DIRS}
    )
    
    target_compile_definitions(stitch_c_interface PRIVATE
        CAMERA_MANAGER_EXPORTS
    )
    
    # 创建C可执行文件
    add_executable(stitch_app stitch_shared.c)
    set_source_files_properties(stitch_shared.c PROPERTIES LANGUAGE C)
    
    target_include_directories(stitch_app PRIVATE
        ${CMAKE_SOURCE_DIR}/inc
    )
    
    target_link_libraries(stitch_c_interface PRIVATE
        camera_manager
        Qt5::Widgets
        qt
        spdlog::spdlog
        ${CUDA_LIBRARIES}
    )
    
    # C程序链接
    target_link_libraries(stitch_app PRIVATE
        stitch_c_interface
        camera_manager
        Qt5::Widgets
        qt
        spdlog::spdlog
        ${CUDA_LIBRARIES}
        stdc++fs
        -Wl,--start-group
        -lstdc++
        -lm
        -Wl,--end-group
    )
    
else()
    message(STATUS "Building C++ version executable")
    
    # 创建C++可执行文件
    add_executable(stitch_app main.cpp)
    
    target_include_directories(stitch_app PRIVATE
        ${CMAKE_SOURCE_DIR}/inc
    )
    
    # C++程序链接
    target_link_libraries(stitch_app PRIVATE
        camera_manager
        Qt5::Widgets
        qt
        spdlog::spdlog
        ${CUDA_LIBRARIES}
    )
endif()

# ========== 通用配置 ==========
if(ENABLE_KERNEL_TEST)
    target_compile_definitions(camera_manager PRIVATE KERNEL_TEST)
endif()

target_link_directories(stitch_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/components/qt/include
)

# 设置输出到bin目录
set_target_properties(stitch_app PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 资源复制
file(COPY ${CMAKE_SOURCE_DIR}/resource
     DESTINATION ${CMAKE_BINARY_DIR}
     FILES_MATCHING PATTERN "*")
file(COPY ${CMAKE_SOURCE_DIR}/scripts
     DESTINATION ${CMAKE_BINARY_DIR}
     FILES_MATCHING PATTERN "*")