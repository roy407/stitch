#pragma once

#include <pthread.h>
#include <stdint.h>
#include <stddef.h>
#include <atomic>

// 定义共享内存名称
static const char* DEFAULT_SHM_NAME = "/stitch_view_shm";

// 20K分辨率图很大 (20800*2160*3 ~ 135MB)
// 6 帧缓冲区约需 810MB 内存。确保机器内存充足。
// 使用 Ring Buffer 模式，BUFFER_COUNT 越大，容忍的消费端抖动越强。
static const size_t SHM_BUFFER_SIZE = 1024 * 1024 * 1024; // 1GB
static const int BUFFER_COUNT = 6; 

struct ShmSlotHeader {
    uint32_t width;
    uint32_t height;
    int pixel_format; // AVPixelFormat
    size_t data_size;
    uint64_t timestamp;
    uint64_t frame_sequence;
};

struct ShmContextHeader {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    bool initialized;
    
    // 循环队列管理
    // write_index: 生产者下一个要写入的位置 (0 ~ N-1)
    // oldest_index: 当前缓冲区中最旧的一帧位置 (用于消费者判断范围)
    volatile int write_index; 
    volatile int oldest_index;
    
    // 每个槽位数据区的起始偏移量
    size_t slot_data_offsets[BUFFER_COUNT];
    // 每个槽位分配的最大容量
    size_t slot_capacity;
};
