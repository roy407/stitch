#include <vector>
#include <mutex>
#include <atomic>
#include <iostream>
#include <queue>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

#include "camera_manager.h"

 //safe_queue<Frame>& launch_stitch_worker() {
list_queue<Frame>& launch_stitch_worker() {
    static camera_manager camera;
    camera.start();
    // 返回引用，立即返回
    return camera.getStitchCameraStream();
}