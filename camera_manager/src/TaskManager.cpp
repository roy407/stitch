#include "TaskManager.h"
#include <iostream>

TaskManager::TaskManager() {

}

TaskManager::~TaskManager() {

}

void TaskManager::start() {
    running = true;
    m_thread = std::thread(&TaskManager::run,this);
}

void TaskManager::stop() {
    running = false;
    if(m_thread.joinable()) {
        m_thread.join();
    }
}

// bool TaskManager::init(void *initMessage)
// {
//     return false;
// }

