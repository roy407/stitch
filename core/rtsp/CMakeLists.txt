add_library(rtsp STATIC
  src/rtsp.cpp
)

link_directories(${FFMPEG_LIB_DIR})

target_include_directories(rtsp
  PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../safe_tools
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${FFMPEG_INCLUDE_DIR}
)

set(SHARE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/share")
set(SHARE_DEST_DIR "${CMAKE_BINARY_DIR}")

file(GLOB_RECURSE SHARE_FILES CONFIGURE_DEPENDS
    "${SHARE_SRC_DIR}/*"
)

add_custom_target(CopyRtspShare ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SHARE_DEST_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${SHARE_SRC_DIR}"
        "${SHARE_DEST_DIR}"
    DEPENDS ${SHARE_FILES}
)