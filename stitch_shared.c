// stitch_shared.c
#include "manage_api.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char* argv[]) {
    printf("=== stitch_shared.c started ===\n");
    
    // 检查命令行参数
    if (argc < 2) {
        printf("Usage: %s <config_file>\n", argv[0]);
        printf("Example: %s resource/cam5.json\n", argv[0]);
        return 1;
    }
    
    const char* config_file = argv[1];
    printf("Using config file: %s\n", config_file);
    
    // 1. 创建句柄
    camera_manger_handle* handle = api_handle_create();
    if (!handle) {
        printf("ERROR: Failed to create handle\n");
        return 1;
    }
    printf("SUCCESS: Handle created at %p\n", (void*)handle);
    
    // 2. 设置配置文件名（必须在初始化之前设置）
    printf("Setting config file...\n");
    int result = camera_manager_set_config_filename(handle, config_file);
    if (result != 0) {
        printf("ERROR: Failed to set config file: %s (error: %d)\n", config_file, result);
        api_handle_destroy(handle);
        return 1;
    }
    printf("SUCCESS: Config file set to: %s\n", config_file);
    
    // 3. 初始化（此时config会自动从文件加载）
    printf("Initializing camera manager...\n");
    result = camera_manager_init_instance(handle);
    printf("camera_manager_init_instance returned: %d\n", result);
    if (result != 0) {
        printf("ERROR: Failed to initialize camera manager\n");
        api_handle_destroy(handle);
        return 1;
    }
    
    // 4. 获取流数量
    size_t stream_count = camera_manager_get_stream_count(handle);
    printf("Stream count: %zu\n", stream_count);
    
    // 5. 启动
    printf("Starting camera manager...\n");
    result = camera_manager_start(handle);
    printf("camera_manager_start returned: %d\n", result);
    if (result != 0) {
        printf("WARNING: Failed to start camera manager\n");
    }
    
    // 6. 等待一段时间（如果需要）
    printf("Running for 15 seconds...\n");
    sleep(15);  // 等待5秒，或者使用其他逻辑
    
    // 7. 停止
    printf("Stopping camera manager...\n");
    result = camera_manager_stop(handle);
    printf("camera_manager_stop returned: %d\n", result);
    if (result != 0) {
        printf("WARNING: Failed to stop camera manager\n");
    }
    
    // 8. 销毁
    printf("Destroying handle...\n");
    api_handle_destroy(handle);
    printf("SUCCESS: Handle destroyed\n");
    
    printf("=== stitch_shared.c finished ===\n");
    return 0;
}