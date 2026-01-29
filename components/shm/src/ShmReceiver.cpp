#include "ShmReceiver.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <errno.h>
#include <time.h>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/mem.h>
}

ShmReceiver::ShmReceiver(const std::string& shm_name, size_t size)
    : m_shm_name(shm_name), m_size(size), m_shm_fd(-1), m_ptr(MAP_FAILED), m_context(nullptr), m_last_sequence(0), m_next_expected_seq(0) {
}

ShmReceiver::~ShmReceiver() {
    close();
}
bool ShmReceiver::init() {
    m_shm_fd = shm_open(m_shm_name.c_str(), O_RDWR, 0666);
    if (m_shm_fd == -1) {
        // Log?
        return false;
    }

    m_ptr = mmap(0, m_size, PROT_READ | PROT_WRITE, MAP_SHARED, m_shm_fd, 0);
    if (m_ptr == MAP_FAILED) {
        std::cerr << "[ShmReceiver] mmap failed: " << strerror(errno) << std::endl;
        ::close(m_shm_fd);
        m_shm_fd = -1;
        return false;
    }

    uint8_t* base = static_cast<uint8_t*>(m_ptr);
    m_context = reinterpret_cast<ShmContextHeader*>(base);
    
    int retry = 0;
    while (!m_context->initialized && retry < 100) {
        usleep(10000); 
        retry++;
    }
    if (!m_context->initialized) {
        return false;
    }

    size_t offset = sizeof(ShmContextHeader);
    for (int i = 0; i < BUFFER_COUNT; ++i) {
        m_slots[i] = reinterpret_cast<ShmSlotHeader*>(base + offset);
        offset += sizeof(ShmSlotHeader);
    }
    
    if (offset % 64 != 0) {
        offset += (64 - (offset % 64));
    }
    
    for (int i = 0; i < BUFFER_COUNT; ++i) {
        m_data_ptrs[i] = base + offset;
        offset += m_context->slot_capacity;
    }

    // Ref count
    pthread_mutex_lock(&m_context->mutex);
    m_context->ref_count++;
    m_attached = true;
    pthread_mutex_unlock(&m_context->mutex);

    return true;
}

bool ShmReceiver::recvFrame(AVFrame* frame, int timeout_ms) {
    if (!m_ptr || !m_context || !frame) return false;

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += timeout_ms / 1000;
    ts.tv_nsec += (timeout_ms % 1000) * 1000000;
    if (ts.tv_nsec >= 1000000000) {
        ts.tv_sec += 1;
        ts.tv_nsec -= 1000000000;
    }

    pthread_mutex_lock(&m_context->mutex);

    // Ring Buffer 读取逻辑：
    // 我们希望读到 >= m_next_expected_seq 的最早一帧。
    // 如果 m_next_expected_seq 比较老（被 overwrite了），则只能被迫跳到 oldest。
    
    int target_slot = -1;
    
    while (true) {
        int oldest = m_context->oldest_index;
        int write = m_context->write_index;
        
        // 1. 判断是否为空
        // 如果 oldest == write 且 seq == 0 ?? 初始状态
        // 简单判断: 检查 oldest 的 seq 是否 > m_last_sequence
        // 或者: 检查是否有数据可读
        
        // 缓冲区中的有效数据范围是从 oldest 到 (write - 1)
        // 我们遍历这个区间 (逻辑上的环形区间)
        
        int count = (write - oldest + BUFFER_COUNT) % BUFFER_COUNT;
        
        // 注意：如果 ring buffer 满了，write == oldest ? 
        // 在 Sender 逻辑里，如果满就是说 f了，Oldest 会被推走，所以 write 永远追不上 oldest 
        // 也ull 的时候，write == oldest 是不可能的，offset 始终保持 1 ?
        // 不，m_context->write_index 指向下一个*空*位。
        // 如果 write == oldest，说明 buffer 是空的！
        // 因为 Sender 写完会把 write 移走。如果满，oldest 也会移走。
        // 所以 write == oldest 意味着 EMPTY.
        
        if (write == oldest) {
             // Buffer Empty, Wait.
        } else {
             // 2. 寻找合适的帧
             // 我们从 oldest 开始找
             int curr = oldest;
             while (curr != write) {
                 if (m_slots[curr]->frame_sequence > m_last_sequence) {
                     target_slot = curr;
                     goto found;
                 }
                 curr = (curr + 1) % BUFFER_COUNT;
             }
        }

        // Wait
        int ret = pthread_cond_timedwait(&m_context->cond, &m_context->mutex, &ts);
        if (ret == ETIMEDOUT) {
            pthread_mutex_unlock(&m_context->mutex);
            return false;
        }
    }
    
found:
    // 找到了 target_slot
    // 这个时候还在 Lock 里面
    
    // 检查是否发生了跳帧
    uint64_t found_seq = m_slots[target_slot]->frame_sequence;
    if (m_next_expected_seq > 0 && found_seq > m_next_expected_seq) {
        // 警告：发生了丢帧 (Overwritten or Consumer too slow)
        // std::cerr << "Frame drop! Expected " << m_next_expected_seq << " but got " << found_seq << std::endl;
    }
    
    // 更新期望
    m_next_expected_seq = found_seq + 1;
    m_last_sequence = found_seq;
    
    pthread_mutex_unlock(&m_context->mutex);
    
    // Copy Data (Read-only, safe as long as writer doesn't lap us)
    // 只要 Writer 不要写一圈回来覆盖这个 target_slot 就安全。
    // 考虑到 135MB copy 时间 vs 135MB*6 fill 时间，
    // Writer 写满 6 帧需要的时间通常远大于 Reader读一帧的时间。
    // 除非 Reader 极其慢而 Writer 极其快。
    
    ShmSlotHeader* header = m_slots[target_slot];
    frame->width = header->width;
    frame->height = header->height;
    frame->format = header->pixel_format;
    frame->pts = header->timestamp;

    if (av_frame_get_buffer(frame, 32) < 0) {
        return false;
    }
    
    uint8_t* src_data[4];
    int src_linesize[4];
    av_image_fill_arrays(src_data, src_linesize, m_data_ptrs[target_slot], 
                         (AVPixelFormat)frame->format, frame->width, frame->height, 1);
    
    av_image_copy(frame->data, frame->linesize, (const uint8_t**)src_data, src_linesize, 
                  (AVPixelFormat)frame->format, frame->width, frame->height);

    return true;
}

void ShmReceiver::close() {
    if (m_ptr != MAP_FAILED) {
        bool should_unlink = false;
        if (m_context && m_attached) {
            pthread_mutex_lock(&m_context->mutex);
            m_context->ref_count--;
            if (m_context->ref_count <= 0) {
                should_unlink = true;
            }
            pthread_mutex_unlock(&m_context->mutex);
            m_attached = false;
        }

        munmap(m_ptr, m_size);
        m_ptr = MAP_FAILED;
        
        if (should_unlink) {
             shm_unlink(m_shm_name.c_str());
             std::cout << "[ShmReceiver] Last user, unlinked shm: " << m_shm_name << std::endl;
        }
    }
    if (m_shm_fd != -1) {
        ::close(m_shm_fd);
        m_shm_fd = -1;
    }
}
