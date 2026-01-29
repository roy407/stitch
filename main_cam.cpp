#include "camera_manager.h"
#include "config.h"
#include "safe_queue.hpp" // Not needed directly, but keeping it consistent if included implicitly
#include <iostream>
#include <thread>
#include <string>
#include <signal.h>

bool running = true;

void signal_handler(int sig) {
    running = false;
}

int main(int argc, char *argv[]) {
    std::string config_name = "";
    if (argc > 1) {
        config_name = argv[1];
    } else {
        std::cerr << "Usage: ./stitch_cam <config.json>" << std::endl;
        return -1;
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    config::SetConfigFileName(config_name);
    
    std::cout << "[Backend] Initializing Camera Manager..." << std::endl;
    // 启动相机管理器（内部会自动初始化SHM Sender）
    camera_manager* cam = camera_manager::GetInstance();
    cam->start();
    
    std::cout << "[Backend] Stitch Camera Service Started. Waiting for connections..." << std::endl;
    
    // 阻塞主线程
    while(running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "[Backend] Stopping..." << std::endl;
    cam->stop();
    return 0;
}
