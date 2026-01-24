file(GLOB SHM_SOURCES src/*.cpp)
add_library(shm_lib STATIC ${SHM_SOURCES})
target_include_directories(shm_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(shm_lib PUBLIC avutil pthread rt)

