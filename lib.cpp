#include <vector>
#include <mutex>
#include <atomic>
#include <iostream>
#include <queue>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

#include "camera_manager.h"

safe_queue<T_Frame>& launch_stitch_worker() {
    static camera_manager camera;

    // 启动线程运行 camera.start()（只启动一次）
    static std::once_flag flag;
    std::call_once(flag, []() {
        std::thread([]() {
            camera.start();  // 阻塞函数放在线程中
        }).detach();
    });

    // 返回引用，立即返回
    return camera.get_stitch_stream();
}