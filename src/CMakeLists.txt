# src/CMakeLists.txt

# 添加头文件搜索路径
include_directories(
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_SOURCE_DIR}/core/config/include
    ${CMAKE_SOURCE_DIR}/camera_manager/include # 添加camera_manager头文件路径
)

# 设置源代码文件
set(SOURCES
    cpp_exposure_to_c.cpp
)

# 设置头文件
set(HEADERS
    ${CMAKE_SOURCE_DIR}/inc/manage_api.h
    ${CMAKE_SOURCE_DIR}/core/config/include
)

# 创建C接口库
add_library(stitch_lib SHARED ${SOURCES} ${HEADERS})
add_library(stitch::stitch ALIAS stitch_lib)

# 设置库属性
set_target_properties(stitch_lib PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "stitch"
    PUBLIC_HEADER "${CMAKE_SOURCE_DIR}/inc/manage_api.h"
)

# 添加编译定义 - 重要！
target_compile_definitions(stitch_lib PRIVATE
    STITCH_EXPORTS
    CAMERA_MANAGER_EXPORTS  # 确保这个被定义
)

# 设置C++标准
target_compile_features(stitch_lib PRIVATE cxx_std_17)

# 链接必要的库
target_link_libraries(stitch_lib PRIVATE
    camera_manager      # 链接camera_manager库
    Qt5::Widgets
    spdlog::spdlog
    ${CUDA_LIBRARIES}
    # 其他需要的库...
)

# 安装配置（可选）
install(TARGETS stitch_lib
    EXPORT stitch-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include
)

install(EXPORT stitch-targets
    FILE stitch-config.cmake
    NAMESPACE stitch::
    DESTINATION lib/cmake/stitch
)