#include "stitch.h"
#include "camera_manager.h"
#include <string.h>

static camera_manager* __cam_handle__ = nullptr;

#ifdef __cplusplus
extern "C" {
#endif

CAMERA_MANAGER_API int camera_manager_init_instance(const char* filename) {

}

CAMERA_MANAGER_API int camera_manager_start(void) {

}

CAMERA_MANAGER_API int camera_manager_stop(void) {

}

CAMERA_MANAGER_API int camera_manager_set_stitch_callback(
    int pipeline_id, 
    camera_callback callback
) {

}

CAMERA_MANAGER_API int camera_manager_set_camera_callback(
    int cam_id, 
    camera_callback callback
) {

}

CAMERA_MANAGER_API size_t camera_manager_get_stream_count(void) {

}

#ifdef __cplusplus
}
#endif
