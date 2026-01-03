#ifndef CAMERA_MANAGER_API_H
#define CAMERA_MANAGER_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stitch_types.h"

#define CAMERA_MANAGER_API __attribute__((visibility("default")))

typedef void (*camera_callback)(stitch_frame_t data);

/**
 * @brief 初始化相机管理器实例
 * @return 成功返回1，失败返回0
 */
CAMERA_MANAGER_API int camera_manager_init_instance(const char* filename);

/**
 * @brief 启动相机管理器
 * @return 成功返回1，失败返回0
 */
CAMERA_MANAGER_API int camera_manager_start(void);

/**
 * @brief 停止相机管理器
 * @return 成功返回1，失败返回0
 */
CAMERA_MANAGER_API int camera_manager_stop(void);

/**
 * @brief 设置拼接图回调函数
 * @param pipeline_id 管道ID
 * @param callback 回调函数指针
 * @return 成功返回1，失败返回0
 */
CAMERA_MANAGER_API int camera_manager_set_stitch_callback(
    int pipeline_id, 
    camera_callback callback
);

/**
 * @brief 设置单个相机流回调函数
 * @param cam_id 相机ID
 * @param callback 回调函数指针
 * @return 成功返回1，失败返回0
 */
CAMERA_MANAGER_API int camera_manager_set_camera_callback(
    int cam_id, 
    camera_callback callback
);

/**
 * @brief 获取相机流数量
 * @return 相机流数量，失败返回0
 */
CAMERA_MANAGER_API size_t camera_manager_get_stream_count(void);

#ifdef __cplusplus
}
#endif

#endif // CAMERA_MANAGER_API_H