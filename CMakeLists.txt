# /home/eric/stitch/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(pstitch CUDA CXX)

# 在 project() 后添加
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 用于检测野指针问题，但是打开后会导致cuInit创建失败，显示OUT_OF_MEMORY
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -g")
# 这个问题是 已知现象，原因不是你真的"内存不够"，而是 ASan 与 CUDA（尤其是 NVIDIA 驱动的 
# libcuda.so, libnvidia-fatbinaryloader.so, libnpp*, libnvrtc 等） 在运行时存在冲突。

option(BUILD_SHARED_LIB "Build as shared library instead of executable" OFF)
option(ENABLE_KERNEL_TEST "Enable kernel test features" OFF)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(spdlog REQUIRED)

# 设置ffmpeg的配置
set(FFMPEG_INCLUDE_DIRS "/usr/local/include")

# 定义公共包含目录
set(COMMON_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_SOURCE_DIR}/camera_manager/include
    ${CMAKE_SOURCE_DIR}/components/qt/include
    ${CUDA_INCLUDE_DIRS}
)

add_executable(stitch_app main.cpp)

set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(core)
add_subdirectory(components)
add_subdirectory(camera_manager)

if(ENABLE_KERNEL_TEST)
    target_compile_definitions(camera_manager PRIVATE KERNEL_TEST)
endif()

# 为 stitch_app 添加包含目录
target_include_directories(stitch_app PRIVATE ${COMMON_INCLUDE_DIRS})

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
    set_target_properties(stitch_app PROPERTIES POSITION_INDEPENDENT_CODE ON)
    
    message(STATUS "Building C interface shared library: libstitch.so")
    
    # 构建为共享库
    add_library(stitch_shared SHARED cpp_exposure_to_c.cpp)
    
    # 确保所有依赖符号正确导出
    target_compile_definitions(stitch_shared PRIVATE
        STITCH_EXPORTS
        QT_CORE_LIB
        QT_GUI_LIB
        QT_WIDGETS_LIB
    )
    
    
    target_compile_options(stitch_shared PRIVATE
        -fvisibility=hidden
        -fvisibility-inlines-hidden
    )
    
    # 设置库属性
    set_target_properties(stitch_shared PROPERTIES
        VERSION 1.0.0
        SOVERSION 1
        OUTPUT_NAME "VisionEngine"
        POSITION_INDEPENDENT_CODE ON
        # 设置 RPATH 以便在其他位置也能找到依赖
        INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
        BUILD_WITH_INSTALL_RPATH TRUE
        MACOSX_RPATH ON
    )
    
    # 包含目录
    target_include_directories(stitch_shared PRIVATE ${COMMON_INCLUDE_DIRS})
    
    # 链接库 - 添加所有必要的依赖
    target_link_libraries(stitch_shared PRIVATE 
        camera_manager
        Qt5::Widgets
        Qt5::Core
        Qt5::Gui
        qt
        spdlog::spdlog
    )

    # 安装动态库
    install(TARGETS stitch_shared LIBRARY DESTINATION lib)
    # 安装头文件（如果有的话）
    install(DIRECTORY inc/ DESTINATION include)
endif()

file(COPY ${CMAKE_SOURCE_DIR}/resource
     DESTINATION ${CMAKE_BINARY_DIR}
     FILES_MATCHING PATTERN "*")
file(COPY ${CMAKE_SOURCE_DIR}/scripts
     DESTINATION ${CMAKE_BINARY_DIR}
     FILES_MATCHING PATTERN "*")