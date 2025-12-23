#include "TaskManager.h"
#include <iostream>
#include "log.hpp"

TaskManager::TaskManager() {
    LOG_DEBUG("TaskManager::TaskManager");
}

TaskManager::~TaskManager() {

}

void TaskManager::start() {
    running = true;
    m_thread = std::thread(&TaskManager::run,this);
    thread_cnt ++;
    LOG_DEBUG("thread {} created",m_name);
}

void TaskManager::stop() {
    running = false;
    if(m_thread.joinable()) {
        m_thread.join();
    }
    thread_cnt --;
    LOG_DEBUG("thread {} exit! left {} threads has not exited .",m_name, thread_cnt);
}

// bool TaskManager::init(void *initMessage)
// {
//     return false;
// }

uint64_t TaskManager::thread_cnt = 0;

