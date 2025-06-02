add_library(config STATIC
    src/config.cpp
)

target_include_directories(config
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(nlohmann_json REQUIRED)
target_link_libraries(config
    PUBLIC nlohmann_json::nlohmann_json
)
