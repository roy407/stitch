#include <vector>
#include <mutex>
#include <atomic>
#include <iostream>
#include <queue>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

#include "camera_manager.h"

int main() {
    camera_manager camera;
    camera.start();
    std::cout<<__func__<<" exit!"<<std::endl;
    return 0;
}