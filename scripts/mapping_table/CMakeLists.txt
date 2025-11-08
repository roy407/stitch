cmake_minimum_required(VERSION 3.16)
project(compute_mapping_table LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/config/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/utils/include
)

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/config
    ${CMAKE_BINARY_DIR}/core_config_build     # 二进制构建目录
)

add_executable(compute_mt
    compute_mapping_table.cpp
)

target_link_libraries(compute_mt PRIVATE config)
