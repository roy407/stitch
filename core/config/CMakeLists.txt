add_library(config STATIC
    src/config.cpp
)

target_include_directories(config PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CORE_TOP_DIR}/utils/include
)

find_package(nlohmann_json REQUIRED)
target_link_libraries(config PRIVATE
    nlohmann_json::nlohmann_json
)
