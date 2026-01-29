#pragma once

#include "ShmCommon.h"
#include <string>

struct AVFrame;

class ShmReceiver {
public:
    ShmReceiver(const std::string& shm_name = DEFAULT_SHM_NAME, size_t size = SHM_BUFFER_SIZE);
    ~ShmReceiver();

    bool init();
    // 阻塞等待并读取一帧数据
    bool recvFrame(AVFrame* destination_frame, int timeout_ms = 1000);
    
    void close();

private:
    std::string m_shm_name;
    size_t m_size;
    int m_shm_fd;
    void* m_ptr;
    
    ShmContextHeader* m_context;
    ShmSlotHeader* m_slots[BUFFER_COUNT];
    uint8_t* m_data_ptrs[BUFFER_COUNT];
    
    uint64_t m_last_sequence; 
    
    // 消费者记录自己想要读的下一帧 SEQ
    // 用于确保连续性
    uint64_t m_next_expected_seq;
    bool m_attached = false;
};
