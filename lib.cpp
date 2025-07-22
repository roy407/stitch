#include <vector>
#include <mutex>
#include <atomic>
#include <iostream>
#include <queue>
#include <chrono>

#include "camera_manager.h"

safe_queue<std::pair<AVFrame*,costTimes>>& launch_stitch_worker() {
    camera_manager* camera = camera_manager::GetInstance();
    camera->start();

    return camera.get_stitch_stream();
}

bool destory_stitch_worker() {
    camera_manager* camera = camera_manager::GetInstance();
    camera->stop();
}