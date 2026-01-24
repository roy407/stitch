#pragma once

#include "ShmCommon.h"
#include <string>

struct AVFrame;

class ShmSender {
public:
    ShmSender(const std::string& shm_name = DEFAULT_SHM_NAME, size_t size = SHM_BUFFER_SIZE);
    ~ShmSender();

    bool init();
    // 将 AVFrame 数据写入共享内存
    bool sendFrame(const AVFrame* frame);
    // 释放资源
    void close();

private:
    std::string m_shm_name;
    size_t m_size;
    int m_shm_fd;
    void* m_ptr;
    
    ShmContextHeader* m_context;
    ShmSlotHeader* m_slots[BUFFER_COUNT];
    uint8_t* m_data_ptrs[BUFFER_COUNT];
};
