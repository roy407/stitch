// stitchAPI.cpp
#include <iostream>
#include <string>
#include "StitchAPI.h"
#include "camera_manager.h"
#include "CallbackConsumer.h"
#include "log.hpp"
#include "config.h"

// 结构体定义
struct camera_manger_handle {
    int initialized;          // 防止重复初始化
    camera_manager* cam_handle;
};

extern "C" {

CAMERA_MANAGER_API camera_manger_handle* api_handle_create(void) {
    camera_manger_handle* handle = new (std::nothrow) camera_manger_handle();
    if (!handle) {
        LOG_ERROR("api_handle_create: init handle failed");
        return nullptr;
    }

    handle->initialized = 0;
    handle->cam_handle  = nullptr;
    LOG_DEBUG("api_handle_create: created handle");
    return handle;
}

CAMERA_MANAGER_API void api_handle_destroy(camera_manger_handle* handle) {
    if (handle) {
        delete handle;
        LOG_DEBUG("api_handle_destroy: handle destroyed");
    } else {
        LOG_WARN("api_handle_destroy: handle already destroyed");
    }
}

CAMERA_MANAGER_API STITCH_STATUS
camera_manager_init_instance(camera_manger_handle* handle) {
    if (!handle) {
        LOG_ERROR("camera_manager_init_instance: invalid handle");
        return STITCH_ERR_INVALID;
    }

    camera_manager* cam = camera_manager::GetInstance();
    if (!cam) {
        return STITCH_ERR_INVALID;
    }

    handle->initialized = 1;
    handle->cam_handle  = cam;
    LOG_DEBUG("camera_manager_init_instance: initialized handle");
    return STITCH_OK;
}

CAMERA_MANAGER_API STITCH_STATUS
camera_manager_set_config_filename(camera_manger_handle* handle,
                                   const char* cfg_name) {
    if (!handle || !cfg_name) {
        LOG_ERROR("camera_manager_set_config_filename: invalid handle or filename");
        return STITCH_ERR_INVALID;
    }

    try {
        config::SetConfigFileName(std::string(cfg_name));
        LOG_DEBUG("camera_manager_set_config_filename: set config to {}", cfg_name);
        return STITCH_OK;
    } catch (const std::exception& e) {
        LOG_ERROR("Error setting config filename: {}", e.what());
        return STITCH_ERR_CONFIG;
    }
}

CAMERA_MANAGER_API const char*
camera_manager_get_config_filename(camera_manger_handle* handle) {
    if (!handle) {
        LOG_ERROR("camera_manager_get_config_filename: invalid handle");
        return nullptr;
    }

    static std::string config_name;
    try {
        config_name = config::GetConfigFileName();
        return config_name.c_str();
    } catch (const std::exception& e) {
        LOG_ERROR("Error getting config filename: {}", e.what());
        return nullptr;
    }
}

CAMERA_MANAGER_API STITCH_STATUS
camera_manager_reload_config(camera_manger_handle* handle) {
    if (!handle) {
        return STITCH_ERR_INVALID;
    }

    try {
        std::string cfg_name = config::GetConfigFileName();
        if (cfg_name.empty()) {
            LOG_ERROR("No config file name set");
            return STITCH_ERR_CONFIG;
        }

        LOG_DEBUG("camera_manager_reload_config: reloading config from {}", cfg_name);
        return STITCH_OK;
    } catch (const std::exception& e) {
        LOG_ERROR("Error reloading config: {}", e.what());
        return STITCH_ERR_RELOAD;
    }
}

CAMERA_MANAGER_API size_t
camera_manager_get_stream_count(camera_manger_handle* handle) {
    if (!handle || !handle->initialized) {
        return 0;
    }

    if (handle->cam_handle) {
        return handle->cam_handle->getCameraStreamCount();
    }

    return 0;
}

CAMERA_MANAGER_API STITCH_STATUS
camera_manager_start(camera_manger_handle* handle) {
    if (!handle || !handle->initialized) {
        return STITCH_ERR_INVALID;
    }

    if (handle->cam_handle) {
        handle->cam_handle->start();
        LOG_DEBUG("camera_manager_start: started");
        return STITCH_OK;
    }

    return STITCH_ERR_INVALID;
}

CAMERA_MANAGER_API STITCH_STATUS
camera_manager_stop(camera_manger_handle* handle) {
    if (!handle || !handle->initialized) {
        return STITCH_ERR_INVALID;
    }

    if (handle->cam_handle) {
        handle->cam_handle->stop();
        LOG_DEBUG("camera_manager_stop: stopped");
        return STITCH_OK;
    }

    return STITCH_ERR_INVALID;
}

CAMERA_MANAGER_API STITCH_STATUS
camera_manager_set_stitch_callback(camera_manger_handle* handle,
                                   int pipeline_id,
                                   stitch_callback_t callback) {
    if (!handle || !handle->initialized) {
        return STITCH_ERR_INVALID;
    }

    auto wrapper = [callback](Frame internal_frame) {
        callback(internal_frame);
    };

    handle->cam_handle->setStitchStreamCallback(pipeline_id, wrapper);
    LOG_DEBUG("camera_manager_set_stitch_callback: callback set for pipe {}", pipeline_id);
    return STITCH_OK;
}

CAMERA_MANAGER_API STITCH_STATUS
camera_manager_set_camera_callback(camera_manger_handle* handle,
                                   int cam_id,
                                   stitch_callback_t callback) {
    if (!handle || !handle->initialized) {
        return STITCH_ERR_INVALID;
    }

    auto wrapper = [callback](Frame internal_frame) {
        callback(internal_frame);
    };

    handle->cam_handle->setCameraStreamCallback(cam_id, wrapper);
    LOG_DEBUG("camera_manager_set_camera_callback: callback set for camera {}", cam_id);
    return STITCH_OK;
}

} // extern "C"
