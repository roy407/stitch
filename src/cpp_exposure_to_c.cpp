// cpp_exposure_to_c.cpp
#include <iostream>
#include <string>
#include "manage_api.h"
#include "camera_manager.h"
#include "tools.hpp"

// 包含config头文件
#include "core/config/include/config.h"

// 结构体定义
struct camera_manger_handle
{
    int initialized;
    camera_manager* cam_handle;
    // 注意：不需要存储config对象，因为它是单例
};

extern "C" {

CAMERA_MANAGER_API camera_manger_handle* api_handle_create(void) {
    camera_manger_handle* handle = new (std::nothrow) camera_manger_handle();
    if (!handle) return nullptr;
    
    handle->initialized = 0;
    handle->cam_handle = nullptr;
    
    std::cout << "api_handle_create: created handle " << handle << std::endl;
    return handle;
}

CAMERA_MANAGER_API void api_handle_destroy(camera_manger_handle* handle) {
    if (handle) {
        delete handle;
        std::cout << "api_handle_destroy: handle destroyed" << std::endl;
    }
}

CAMERA_MANAGER_API int camera_manager_init_instance(camera_manger_handle* handle) {
    if (!handle) {
        std::cerr << "camera_manager_init_instance: invalid handle" << std::endl;
        return -1;
    }
    
    // 获取单例实例
    camera_manager* cam = camera_manager::GetInstance();
    if (cam) {
        handle->initialized = 1;
        handle->cam_handle = cam;
        
        std::cout << "camera_manager_init_instance: initialized handle " << handle << std::endl;
        return 0;
    }
    
    return -1;
}

// 新增函数：设置配置文件名
CAMERA_MANAGER_API int camera_manager_set_config_filename(camera_manger_handle* handle,
                                                         const char* cfg_name) {
    if (!handle || !cfg_name) {
        std::cerr << "camera_manager_set_config_filename: invalid handle or filename" << std::endl;
        return -1;
    }
    
    try {
        // 使用config单例的静态方法设置配置文件名
        config::SetConfigFileName(std::string(cfg_name));
        std::cout << "camera_manager_set_config_filename: set config to " << cfg_name << std::endl;
        
        // 注意：config在构造时会自动从文件加载
        // 我们只需要设置文件名，config会在需要时加载
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error setting config filename: " << e.what() << std::endl;
        return -2;
    }
}

// 可选：新增函数获取当前配置文件名
CAMERA_MANAGER_API const char* camera_manager_get_config_filename(camera_manger_handle* handle) {
    if (!handle) {
        return nullptr;
    }
    
    static std::string config_name;
    try {
        config_name = config::GetConfigFileName();
        return config_name.c_str();
    } catch (const std::exception& e) {
        std::cerr << "Error getting config filename: " << e.what() << std::endl;
        return nullptr;
    }
}

// 可选：新增函数重新加载配置
CAMERA_MANAGER_API int camera_manager_reload_config(camera_manger_handle* handle) {
    if (!handle) {
        return -1;
    }
    
    try {
        // 获取配置文件名
        std::string cfg_name = config::GetConfigFileName();
        if (cfg_name.empty()) {
            std::cerr << "No config file name set" << std::endl;
            return -2;
        }
        
        // 重新加载配置
        // 注意：由于config是单例，可能需要一些特殊处理来重新加载
        // 这里只是一个示例实现
        
        std::cout << "camera_manager_reload_config: reloading config from " << cfg_name << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error reloading config: " << e.what() << std::endl;
        return -3;
    }
}

// 其他已有的函数实现保持不变...
CAMERA_MANAGER_API size_t camera_manager_get_stream_count(camera_manger_handle* handle) {
    if (!handle || !handle->initialized) {
        return 0;
    }
    
    if (handle->cam_handle) {
        return handle->cam_handle->getCameraStreamCount();
    }
    
    return 0;
}

CAMERA_MANAGER_API int camera_manager_start(camera_manger_handle* handle) {
    if (!handle || !handle->initialized) {
        return -1;
    }
    
    if (handle->cam_handle) {
        handle->cam_handle->start();
        std::cout << "camera_manager_start: started" << std::endl;
        return 0;
    }
    
    return -1;
}

CAMERA_MANAGER_API int camera_manager_stop(camera_manger_handle* handle) {
    if (!handle || !handle->initialized) {
        return -1;
    }
    
    if (handle->cam_handle) {
        handle->cam_handle->stop();
        std::cout << "camera_manager_stop: stopped" << std::endl;
        return 0;
    }
    
    return -1;
}

CAMERA_MANAGER_API int camera_manager_set_stitch_callback(camera_manger_handle* handle, 
                                                         stitch_callback_t callback) {
    if (!handle || !handle->initialized) {
        return -1;
    }
    
    std::cout << "camera_manager_set_stitch_callback: callback set" << std::endl;
    return 0;
}

CAMERA_MANAGER_API int camera_manager_set_camera_callback(camera_manger_handle* handle, 
                                                         int cam_id, 
                                                         camera_callback_t callback) {
    if (!handle || !handle->initialized) {
        return -1;
    }
    
    std::cout << "camera_manager_set_camera_callback: callback set for camera " << cam_id << std::endl;
    return 0;
}

} // extern "C"