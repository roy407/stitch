#include "TaskManager.h"
#include <iostream>
#include "log.hpp"

TaskManager::TaskManager() {
    LOG_DEBUG("thread created",m_name);
}

TaskManager::~TaskManager() {
    LOG_DEBUG("thread destoryed",m_name);
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

