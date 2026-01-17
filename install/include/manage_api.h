// manage_api.h
#ifndef MANAGE_API_H
#define MANAGE_API_H

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// 导出宏
#define CAMERA_MANAGER_API __attribute__((visibility("default")))


// 不透明句柄，api是camera_manger的别名，用这个h文件的可以用api，用exposure.cpp的两者都可以用
struct camera_manger_handle;
typedef struct camera_manger_handle api_handle;

// 回调函数类型
typedef void (*stitch_callback_t)(void* data);
// typedef void (*camera_callback_t)(int cam_id, void* data);

/**
 * @brief 创建相机管理器句柄
 * @return 成功返回句柄指针，失败返回NULL
 */
CAMERA_MANAGER_API api_handle* api_handle_create(void);

/**
 * @brief 销毁相机管理器句柄
 * @param handle 句柄指针
 */
CAMERA_MANAGER_API void api_handle_destroy(api_handle* handle);

/**
 * @brief 初始化相机管理器实例
 * @param handle 句柄指针
 * @return 成功返回0，失败返回错误码
 */
CAMERA_MANAGER_API int camera_manager_init_instance(api_handle* handle);

/**
 * @brief 启动相机管理器
 * @param handle 句柄指针
 * @return 成功返回0，失败返回错误码
 */
CAMERA_MANAGER_API int camera_manager_start(api_handle* handle);

/**
 * @brief 停止相机管理器
 * @param handle 句柄指针
 * @return 成功返回0，失败返回错误码
 */
CAMERA_MANAGER_API int camera_manager_stop(api_handle* handle);

/**
 * @brief 设置拼接图回调函数
 * @param handle 句柄指针
 * @param callback 回调函数指针
 * @return 成功返回0，失败返回错误码
 */
CAMERA_MANAGER_API int camera_manager_set_stitch_callback(api_handle* handle, 
                                                         int pipe_id,stitch_callback_t callback);

/**
 * @brief 设置单个相机流回调函数
 * @param handle 句柄指针
 * @param cam_id 相机ID
 * @param callback 回调函数指针
 * @return 成功返回0，失败返回错误码
 */
CAMERA_MANAGER_API int camera_manager_set_camera_callback(api_handle* handle, 
                                                         int cam_id,stitch_callback_t callback);

/**
 * @brief 获取相机流数量
 * @param handle 句柄指针
 * @return 相机流数量，失败返回0
 */
CAMERA_MANAGER_API size_t camera_manager_get_stream_count(api_handle* handle);


/**
 * @brief 设置配置文件名
 * @param handle 句柄指针
 * @param cfg_name 配置文件名
 * @return 成功返回0，失败返回错误码
 */
CAMERA_MANAGER_API int camera_manager_set_config_filename(api_handle* handle,
                                                         const char* cfg_name);

/**
 * @brief 获取当前配置文件名
 * @param handle 句柄指针
 * @return 配置文件名，失败返回NULL
 */
CAMERA_MANAGER_API const char* camera_manager_get_config_filename(api_handle* handle);

/**
 * @brief 重新加载配置文件
 * @param handle 句柄指针
 * @return 成功返回0，失败返回错误码
 */
CAMERA_MANAGER_API int camera_manager_reload_config(api_handle* handle);

#ifdef __cplusplus
}
#endif

#endif // MANAGE_API_H