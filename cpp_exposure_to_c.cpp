// cpp_exposure_to_c.cpp
#include <iostream>
#include <string>
#include "manage_api.h"
#include "stitch_types.h"
#include "camera_manager.h"
#include "CallbackConsumer.h"
#include "tools.hpp"
#include "log.hpp"

// 包含config头文件
#include "config.h"


// 结构体定义
struct camera_manger_handle
{
    int initialized;   //防止重复初始化
    camera_manager* cam_handle;
    // 注意：不需要存储config对象，因为它是单例
};

// 在 C++ 中实现这些结构体
struct types_costTimes {
    uint64_t image_frame_cnt[TYPES_MAX_CAM_SIZE] = {};
    uint64_t when_get_packet[TYPES_MAX_CAM_SIZE] = {};
    uint64_t when_get_decoded_frame[TYPES_MAX_CAM_SIZE] = {};
    uint64_t when_get_stitched_frame = 0;
    uint64_t when_show_on_the_screen = 0;
};

struct types_Frame {
    // 这里可以封装 AVFrame，或者直接使用 AVFrame
    int cam_id = 0;
    AVFrame* m_data = nullptr;
    struct costTimes m_costTimes;
    uint64_t m_timestamp = 0;
};

extern "C" {

CAMERA_MANAGER_API camera_manger_handle* api_handle_create(void) {
    camera_manger_handle* handle = new (std::nothrow) camera_manger_handle();
    if (!handle) return nullptr;
    
    handle->initialized = 0;
    handle->cam_handle = nullptr;
    
    LOG_INFO("api_handle_create: created handle ");
    return handle;
}

CAMERA_MANAGER_API void api_handle_destroy(camera_manger_handle* handle) {
    if (handle) {
        delete handle;
        LOG_INFO("api_handle_destroy: handle destroyed");
    }
}

CAMERA_MANAGER_API int camera_manager_init_instance(camera_manger_handle* handle) {
    if (!handle) {
        LOG_ERROR("camera_manager_init_instance: invalid handle");
        return -1;
    }
    
    // 获取单例实例
    camera_manager* cam = camera_manager::GetInstance();
    if (cam) {
        handle->initialized = 1;
        handle->cam_handle = cam;
        
        LOG_INFO("camera_manager_init_instance: initialized handle ");
        return 0;
    }
    
    return -1;
}

// 新增函数：设置配置文件名
CAMERA_MANAGER_API int camera_manager_set_config_filename(camera_manger_handle* handle,
                                                         const char* cfg_name) {
    if (!handle || !cfg_name) {
        LOG_ERROR("camera_manager_set_config_filename: invalid handle or filename");
        return -1;
    }
    
    try {
        // 使用config单例的静态方法设置配置文件名
        config::SetConfigFileName(std::string(cfg_name));
        LOG_INFO("camera_manager_set_config_filename: set config to ",cfg_name);
        
        // 注意：config在构造时会自动从文件加载
        // 我们只需要设置文件名，config会在需要时加载
        
        return 0;
    } catch (const std::exception& e) {
        LOG_ERROR("Error setting config filename: ",e.what());
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
        LOG_ERROR("Error getting config filename: ",e.what());
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
            LOG_ERROR("No config file name set");
            return -2;
        }
        
        // 重新加载配置
        // 注意：由于config是单例，可能需要一些特殊处理来重新加载
        // 这里只是一个示例实现
        LOG_DEBUG("camera_manager_reload_config: reloading config from ",cfg_name);
        return 0;
    } catch (const std::exception& e) {
        LOG_ERROR("Error reloading config: ",e.what());
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
        LOG_INFO("camera_manager_start: started");
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
        LOG_INFO("camera_manager_stop: stopped");
        return 0;
    }
    
    return -1;
}

// 修改回调设置函数
CAMERA_MANAGER_API int camera_manager_set_stitch_callback(camera_manger_handle* handle, 
                                                         int pipe_id, stitch_callback_t callback) {
    if (!handle || !handle->initialized) {
        return -1;
    }
    
    // 创建一个lambda函数作为包装器
    auto wrapper = [callback](Frame internal_frame) {
        // 将 Frame 转换为 types_Frame_t
        types_Frame_t external_frame;
        
        // 复制基本成员
        external_frame.cam_id = internal_frame.cam_id;
        external_frame.m_data = internal_frame.m_data;
        external_frame.m_timestamp = internal_frame.m_timestamp;
        
        // 复制 costTimes 成员
        for (int i = 0; i < TYPES_MAX_CAM_SIZE; ++i) {
            external_frame.m_costTimes.image_frame_cnt[i] = internal_frame.m_costTimes.image_frame_cnt[i];
            external_frame.m_costTimes.when_get_packet[i] = internal_frame.m_costTimes.when_get_packet[i];
            external_frame.m_costTimes.when_get_decoded_frame[i] = internal_frame.m_costTimes.when_get_decoded_frame[i];
        }
        external_frame.m_costTimes.when_get_stitched_frame = internal_frame.m_costTimes.when_get_stitched_frame;
        external_frame.m_costTimes.when_show_on_the_screen = internal_frame.m_costTimes.when_show_on_the_screen;
        
        // 调用用户提供的回调函数
        if (callback) {
            callback(external_frame);
        }
    };
    
    // 设置回调
    handle->cam_handle->setStitchStreamCallback(pipe_id, wrapper);
    LOG_INFO("camera_manager_set_stitch_callback: callback set for pipe ", pipe_id);
    return 0;
}

CAMERA_MANAGER_API int camera_manager_set_camera_callback(camera_manger_handle* handle, 
                                                         int cam_id,stitch_callback_t callback) {
    if (!handle || !handle->initialized) {
        return -1;
    }
    // 创建一个lambda函数作为包装器
    auto wrapper = [callback](Frame internal_frame) {
        // 将 Frame 转换为 types_Frame_t
        types_Frame_t external_frame;
        
        // 复制基本成员
        external_frame.cam_id = internal_frame.cam_id;
        external_frame.m_data = internal_frame.m_data;
        external_frame.m_timestamp = internal_frame.m_timestamp;
        
        // 复制 costTimes 成员
        for (int i = 0; i < TYPES_MAX_CAM_SIZE; ++i) {
            external_frame.m_costTimes.image_frame_cnt[i] = internal_frame.m_costTimes.image_frame_cnt[i];
            external_frame.m_costTimes.when_get_packet[i] = internal_frame.m_costTimes.when_get_packet[i];
            external_frame.m_costTimes.when_get_decoded_frame[i] = internal_frame.m_costTimes.when_get_decoded_frame[i];
        }
        external_frame.m_costTimes.when_get_stitched_frame = internal_frame.m_costTimes.when_get_stitched_frame;
        external_frame.m_costTimes.when_show_on_the_screen = internal_frame.m_costTimes.when_show_on_the_screen;
        
        // 调用用户提供的回调函数
        if (callback) {
            callback(external_frame);
        }
    };
    handle->cam_handle->setCameraStreamCallback(cam_id,wrapper);
    LOG_INFO("camera_manager_set_camera_callback: callback set for camera ",cam_id);
    return 0;
}

} // extern "C"