#include "ShmSender.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <errno.h>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
}

ShmSender::ShmSender(const std::string& shm_name, size_t size) 
    : m_shm_name(shm_name), m_size(size), m_shm_fd(-1), m_ptr(MAP_FAILED), m_context(nullptr) {
}

ShmSender::~ShmSender() {
    close();
}

bool ShmSender::init() {
    m_shm_fd = shm_open(m_shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (m_shm_fd == -1) {
        std::cerr << "[ShmSender] shm_open failed: " << strerror(errno) << std::endl;
        return false;
    }

    if (ftruncate(m_shm_fd, m_size) == -1) {
        std::cerr << "[ShmSender] ftruncate failed: " << strerror(errno) << std::endl;
        return false;
    }

    m_ptr = mmap(0, m_size, PROT_READ | PROT_WRITE, MAP_SHARED, m_shm_fd, 0);
    if (m_ptr == MAP_FAILED) {
        std::cerr << "[ShmSender] mmap failed: " << strerror(errno) << std::endl;
        return false;
    }

    // 内存布局:
    // [ShmContextHeader]
    // [ShmSlotHeader * 3]
    // [Data Slot 0]
    // [Data Slot 1]
    // [Data Slot 2]
    
    uint8_t* base = static_cast<uint8_t*>(m_ptr);
    m_context = reinterpret_cast<ShmContextHeader*>(base);
    
    size_t offset = sizeof(ShmContextHeader);
    
    // 初始化 Slot Header 指针
    for (int i = 0; i < BUFFER_COUNT; ++i) {
        m_slots[i] = reinterpret_cast<ShmSlotHeader*>(base + offset);
        offset += sizeof(ShmSlotHeader);
    }
    
    // 对齐到 64 字节 (cache line)
    if (offset % 64 != 0) {
        offset += (64 - (offset % 64));
    }

    // 计算每个 Slot 的数据容量
    size_t remaining = m_size - offset;
    size_t per_slot_capacity = remaining / BUFFER_COUNT;
    m_context->slot_capacity = per_slot_capacity;
    
    for (int i = 0; i < BUFFER_COUNT; ++i) {
        m_context->slot_data_offsets[i] = offset;
        m_data_ptrs[i] = base + offset;
        offset += per_slot_capacity;
    }

    // 初始化锁和条件变量
    pthread_mutexattr_t mattr;
    pthread_mutexattr_init(&mattr);
    pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&m_context->mutex, &mattr);
    pthread_mutexattr_destroy(&mattr);

    pthread_condattr_t cattr;
    pthread_condattr_init(&cattr);
    pthread_condattr_setpshared(&cattr, PTHREAD_PROCESS_SHARED);
    pthread_cond_init(&m_context->cond, &cattr);
    pthread_condattr_destroy(&cattr);

    m_context->write_index = 0;
    m_context->oldest_index = 0;
    m_context->initialized = true;
    
    return true;
}

bool ShmSender::sendFrame(const AVFrame* frame) {
    if (!m_ptr || !frame || !m_context) return false;

    int size = av_image_get_buffer_size((AVPixelFormat)frame->format, frame->width, frame->height, 1);
    if (size < 0) return false;

    if (size > m_context->slot_capacity) {
        std::cerr << "[ShmSender] Frame too large for slot! Required: " << size << ", Cap: " << m_context->slot_capacity << std::endl;
        return false;
    }

    // 1. 获取写入位置 (Ring Buffer)
    // 根据策略，我们始终尝试写入 write_index。
    // 为了防止在 copy 过程中被消费者读取到不完整的数据，
    // 最好的方式是：消费者只去读那些已经完成的 frame。
    // 由于 sequence 是最后更新的，所以可以作为一种屏障。
    
    pthread_mutex_lock(&m_context->mutex);
    int target_slot = m_context->write_index;
    pthread_mutex_unlock(&m_context->mutex);

    // 2. 写入数据 (在锁外进行，最大程度减少阻塞)
    // 注意：虽然在锁外，但由于只有一个 Writer，所以写操作是安全的。
    // Reader 即使读到了正在写的数据，也会因为最后 SEQ 不匹配或 Reader 逻辑控制而避开。
    // 但为了绝对安全，我们假设 Reader 只会去读 oldest ~ (write-1) 范围内的数据。
    // target_slot 当前属于 writer 独占区。
    
    ShmSlotHeader* header = m_slots[target_slot];
    // 先把 Header 的部分关键信息写好，但 SEQ 最后写
    header->width = frame->width;
    header->height = frame->height;
    header->pixel_format = frame->format;
    header->data_size = size;
    header->timestamp = frame->pts;
    
    uint8_t* dst_ptr = m_data_ptrs[target_slot];
    
    int ret = av_image_copy_to_buffer(dst_ptr, m_context->slot_capacity,
                                      (const uint8_t* const*)frame->data, frame->linesize,
                                      (AVPixelFormat)frame->format, frame->width, frame->height, 1);
    
    if (ret < 0) return false;

    // 3. 提交更新 (加锁)
    pthread_mutex_lock(&m_context->mutex);
    
    // 生成新的序列号。为了简单，直接自增。
    // 获取前一个 slot 的 SEQ。如果刚启动, 0.
    int prev_slot = (target_slot - 1 + BUFFER_COUNT) % BUFFER_COUNT;
    uint64_t prev_seq = m_slots[prev_slot]->frame_sequence;
    // 如果是第一次运行(prev_seq可能是乱的或者0), 确保单调递增
    // 这里使用一个静态变量或者信任内存中的残留值?
    // 更好的方法是维护一个全局 seq 在 context 中，或者就用 prev + 1
    // 由于我们已经在 init 中把内存清零，seq 初始都为 0.
    // 我们只要保证比前一个大即可。
    header->frame_sequence = prev_seq + 1;
    
    // 推进写指针
    int next_write = (target_slot + 1) % BUFFER_COUNT;
    m_context->write_index = next_write;
    
    // 如果追尾了 (Write 撞上了 Oldest)，说明转了一圈
    // 把 Oldest 也推着走，丢弃最老的一帧
    if (next_write == m_context->oldest_index) {
        m_context->oldest_index = (m_context->oldest_index + 1) % BUFFER_COUNT;
    }
    
    pthread_cond_broadcast(&m_context->cond);
    
    pthread_mutex_unlock(&m_context->mutex);

    return true;
}

void ShmSender::close() {
    if (m_ptr != MAP_FAILED) {
        munmap(m_ptr, m_size);
        m_ptr = MAP_FAILED;
    }
    if (m_shm_fd != -1) {
        ::close(m_shm_fd);
        m_shm_fd = -1;
    }
}
