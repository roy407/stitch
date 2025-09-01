add_library(rtsp STATIC
  src/rtsp.cpp
)

target_include_directories(rtsp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CORE_TOP_DIR}/utils/include
    ${FFMPEG_INCLUDE_DIRS}
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