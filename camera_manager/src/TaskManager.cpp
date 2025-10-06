#include "TaskManager.h"
#include <iostream>
#include "log.hpp"

TaskManager::TaskManager() {
    
}

TaskManager::~TaskManager() {

}

void TaskManager::start() {
    running = true;
    m_thread = std::thread(&TaskManager::run,this);
    LOG_DEBUG("thread {} created",m_name);
}

void TaskManager::stop() {
    running = false;
    if(m_thread.joinable()) {
        m_thread.join();
    }
    LOG_DEBUG("thread {} destoryed",m_name);
}

// bool TaskManager::init(void *initMessage)
// {
//     return false;
// }

