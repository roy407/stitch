add_library(shm STATIC
    src/shm.cpp
)

target_include_directories(shm PRIVATE
    ${CORE_TOP_DIR}/shm/include
    ${CORE_TOP_DIR}/utils/include
    ${CORE_TOP_DIR}/config/include
)

target_link_libraries(shm PRIVATE
    avcodec
    avformat
    avutil
    utils
    config
)



