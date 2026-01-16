// main_dynamic.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>  // 动态加载头文件

// 定义函数指针类型（与manage_api.h中一致）
typedef struct camera_manger_handle camera_manger_handle;

// 函数指针类型定义
typedef camera_manger_handle* (*api_handle_create_func)(void);
typedef void (*api_handle_destroy_func)(camera_manger_handle*);
typedef int (*camera_manager_init_instance_func)(camera_manger_handle*);
typedef int (*camera_manager_start_func)(camera_manger_handle*);
typedef int (*camera_manager_stop_func)(camera_manger_handle*);
typedef size_t (*camera_manager_get_stream_count_func)(camera_manger_handle*);
typedef int (*camera_manager_set_config_filename_func)(camera_manger_handle*, const char*);
typedef const char* (*camera_manager_get_config_filename_func)(camera_manger_handle*);

// 全局函数指针
static void* lib_handle = NULL;
static api_handle_create_func api_handle_create = NULL;
static api_handle_destroy_func api_handle_destroy = NULL;
static camera_manager_init_instance_func camera_manager_init_instance = NULL;
static camera_manager_start_func camera_manager_start = NULL;
static camera_manager_stop_func camera_manager_stop = NULL;
static camera_manager_get_stream_count_func camera_manager_get_stream_count = NULL;
static camera_manager_set_config_filename_func camera_manager_set_config_filename = NULL;
static camera_manager_get_config_filename_func camera_manager_get_config_filename = NULL;

// 加载库函数
int load_stitch_library(const char* lib_path) {
    printf("正在加载库: %s\n", lib_path);
    
    // 打开动态链接库
    lib_handle = dlopen(lib_path, RTLD_LAZY | RTLD_GLOBAL);
    if (!lib_handle) {
        fprintf(stderr, "加载库失败: %s\n", dlerror());
        return -1;
    }
    
    // 清除之前的错误
    dlerror();
    
    // 加载所有函数
    api_handle_create = (api_handle_create_func)dlsym(lib_handle, "api_handle_create");
    api_handle_destroy = (api_handle_destroy_func)dlsym(lib_handle, "api_handle_destroy");
    camera_manager_init_instance = (camera_manager_init_instance_func)dlsym(lib_handle, "camera_manager_init_instance");
    camera_manager_start = (camera_manager_start_func)dlsym(lib_handle, "camera_manager_start");
    camera_manager_stop = (camera_manager_stop_func)dlsym(lib_handle, "camera_manager_stop");
    camera_manager_get_stream_count = (camera_manager_get_stream_count_func)dlsym(lib_handle, "camera_manager_get_stream_count");
    camera_manager_set_config_filename = (camera_manager_set_config_filename_func)dlsym(lib_handle, "camera_manager_set_config_filename");
    camera_manager_get_config_filename = (camera_manager_get_config_filename_func)dlsym(lib_handle, "camera_manager_get_config_filename");
    
    // 检查是否有错误
    char* error = dlerror();
    if (error != NULL) {
        fprintf(stderr, "加载符号失败: %s\n", error);
        dlclose(lib_handle);
        lib_handle = NULL;
        return -2;
    }
    
    // 检查所有必要函数是否都加载成功
    if (!api_handle_create || !api_handle_destroy || !camera_manager_init_instance) {
        fprintf(stderr, "必要函数加载不完整\n");
        dlclose(lib_handle);
        lib_handle = NULL;
        return -3;
    }
    
    printf("库加载成功!\n");
    return 0;
}

// 卸载库函数
void unload_stitch_library() {
    if (lib_handle) {
        dlclose(lib_handle);
        lib_handle = NULL;
        
        // 重置所有函数指针
        api_handle_create = NULL;
        api_handle_destroy = NULL;
        camera_manager_init_instance = NULL;
        camera_manager_start = NULL;
        camera_manager_stop = NULL;
        camera_manager_get_stream_count = NULL;
        camera_manager_set_config_filename = NULL;
        camera_manager_get_config_filename = NULL;
        
        printf("库已卸载\n");
    }
}

// 测试函数
void test_camera_manager() {
    printf("\n=== 开始测试相机管理器 ===\n");
    
    if (!lib_handle) {
        printf("错误: 库未加载\n");
        return;
    }
    
    // 1. 创建句柄
    camera_manger_handle* handle = api_handle_create();
    if (!handle) {
        printf("创建句柄失败!\n");
        return;
    }
    printf("1. 句柄创建成功: %p\n", (void*)handle);
    
    // 2. 设置配置文件
    if (camera_manager_set_config_filename) {
        int ret = camera_manager_set_config_filename(handle, "my_config.json");
        if (ret == 0) {
            printf("2. 配置文件设置成功\n");
        } else {
            printf("2. 配置文件设置失败: %d\n", ret);
        }
    }
    
    // 3. 获取配置文件名
    if (camera_manager_get_config_filename) {
        const char* cfg_name = camera_manager_get_config_filename(handle);
        if (cfg_name) {
            printf("3. 当前配置文件: %s\n", cfg_name);
        }
    }
    
    // 4. 初始化实例
    int ret = camera_manager_init_instance(handle);
    if (ret == 0) {
        printf("4. 初始化成功\n");
    } else {
        printf("4. 初始化失败: %d\n", ret);
    }
    
    // 5. 获取流数量
    if (camera_manager_get_stream_count) {
        size_t count = camera_manager_get_stream_count(handle);
        printf("5. 相机流数量: %zu\n", count);
    }
    
    // 6. 启动
    if (camera_manager_start) {
        ret = camera_manager_start(handle);
        if (ret == 0) {
            printf("6. 启动成功\n");
        } else {
            printf("6. 启动失败: %d\n", ret);
        }
    }
    
    // 7. 停止
    if (camera_manager_stop) {
        ret = camera_manager_stop(handle);
        if (ret == 0) {
            printf("7. 停止成功\n");
        } else {
            printf("7. 停止失败: %d\n", ret);
        }
    }
    
    // 8. 销毁句柄
    api_handle_destroy(handle);
    printf("8. 句柄已销毁\n");
    
    printf("=== 测试完成 ===\n");
}

int main(int argc, char** argv) {
    printf("=== 动态加载测试程序 ===\n");
    
    // 设置库路径
    const char* lib_path = "./lib/libstitch_c_interface.so";
    if (argc > 1) {
        lib_path = argv[1];
    }
    
    // 1. 加载库
    int ret = load_stitch_library(lib_path);
    if (ret != 0) {
        printf("加载库失败, 错误码: %d\n", ret);
        return 1;
    }
    
    // 2. 运行测试
    test_camera_manager();
    
    // 3. 卸载库
    unload_stitch_library();
    
    printf("=== 程序结束 ===\n");
    return 0;
}