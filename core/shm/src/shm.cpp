#include "shm.h"
#include <cstring>
#include <errno.h>
#include <chrono>
#include "log.hpp"

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/pixfmt.h>
}

// 获取当前时间戳（纳秒）
static uint64_t get_now_time() {
    auto now = std::chrono::system_clock::now();
    auto ns_since_epoch = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    ).count();
    return static_cast<uint64_t>(ns_since_epoch);
}


#define MAX_FRAMES 10

StitchCircularBuffer::StitchCircularBuffer() 
    : shm_id_(-1), header_(nullptr), current_frame_(nullptr), is_creator_(false), 
      is_initialized_(false), has_current_data_(false) {}

StitchCircularBuffer::~StitchCircularBuffer() {
    cleanup();
}

// 计算单个帧槽大小 (StitchFrame + Y数据 + UV数据，基于步长)
size_t StitchCircularBuffer::calculate_frame_slot_size(int width, int height, int y_stride, int uv_stride) {
    size_t frame_struct_size = sizeof(StitchFrame);
    // 使用步长计算实际数据大小
    size_t y_data_size = y_stride * height;
    size_t uv_data_size = uv_stride * (height / 2); // NV12格式
    return frame_struct_size + y_data_size + uv_data_size;
}

// 获取下一个循环索引
int StitchCircularBuffer::next_index(int current) {
    return (current + 1) % MAX_FRAMES;
}

// 获取指定索引的帧槽
StitchFrame* StitchCircularBuffer::get_frame_slot(int index) {
    if (!header_ || index < 0 || index >= MAX_FRAMES) {
        return nullptr;
    }
    
    uint8_t* base_ptr = reinterpret_cast<uint8_t*>(header_ + 1); // 跳过头部
    uint8_t* slot_ptr = base_ptr + (index * header_->frame_slot_size);
    return reinterpret_cast<StitchFrame*>(slot_ptr);
}

// 1. 初始化 - 创建10帧循环缓冲区，支持 stride
void StitchCircularBuffer::initialize(const std::string& shm_name, int width, int height, bool create_new, int y_stride, int uv_stride) {
    if (is_initialized_ && header_ != nullptr) {
        return;
    }
    if (is_initialized_ && header_ == nullptr) {
        is_initialized_ = false;
        shm_id_ = -1;
        header_ = nullptr;
        current_frame_ = nullptr;
        has_current_data_ = false;
    }
    shm_name_ = shm_name;
    key_t key = 0;
    for (char c : shm_name) {
        key = key * 31 + c;
    }
    if (create_new) {
        if (width <= 0 || height <= 0 || y_stride <= 0 || uv_stride <= 0) {
            LOG_ERROR("Width, height, y_stride, uv_stride must be specified when creating new shared memory");
            return;
        }
        size_t header_size = sizeof(CircularBufferHeader);
        size_t frame_slot_size = calculate_frame_slot_size(width, height, y_stride, uv_stride);
        shm_size_ = header_size + (frame_slot_size * MAX_FRAMES);
        LOG_WARN("Receiving end creating new shared memory (this should be rare)");
        int old_shm_id = shmget(key, 0, 0);
        if (old_shm_id != -1) {
            shmctl(old_shm_id, IPC_RMID, nullptr);
        }
        shm_id_ = shmget(key, shm_size_, IPC_CREAT | 0666);
        if (shm_id_ == -1) {
            LOG_ERROR("shmget create failed: {}", strerror(errno));
            return;
        }
        is_creator_ = true;
        void* shm_ptr = shmat(shm_id_, nullptr, 0);
        if (shm_ptr == (void*)-1) {
            LOG_ERROR("shmat failed: {}", strerror(errno));
            return;
        }
        header_ = static_cast<CircularBufferHeader*>(shm_ptr);
        memset(header_, 0, shm_size_);
        header_->max_frames = MAX_FRAMES;
        header_->frame_slot_size = frame_slot_size;
        header_->head = 0;
        header_->tail = 0;
        header_->count = 0;
        header_->total_pushed = 0;
        header_->total_popped = 0;
        header_->frames_dropped = 0;
        header_->sequence_counter = 1;
        for (int i = 0; i < MAX_FRAMES; i++) {
            StitchFrame* frame = get_frame_slot(i);
            if (frame) {
                frame->ready = 0;
                frame->width = width;
                frame->height = height;
                frame->y_stride = y_stride;
                frame->uv_stride = uv_stride;
                frame->image_data_size = y_stride * height + uv_stride * (height / 2);
            }
        }
        LOG_INFO("Created circular buffer: {} ({} bytes, {} frames, {} bytes per frame)", 
                 shm_name, shm_size_, MAX_FRAMES, frame_slot_size);
    } else {
        // 接收端：连接到现有共享内存（不需要指定宽高）
        shm_id_ = shmget(key, 0, 0);
        if (shm_id_ == -1) {
            LOG_ERROR("shmget attach failed: {}", strerror(errno));
            return;
        }
        is_creator_ = false;
        struct shmid_ds shm_info;
        if (shmctl(shm_id_, IPC_STAT, &shm_info) == -1) {
            LOG_ERROR("shmctl IPC_STAT failed: {}", strerror(errno));
            return;
        }
        shm_size_ = shm_info.shm_segsz;
        void* shm_ptr = shmat(shm_id_, nullptr, 0);
        if (shm_ptr == (void*)-1) {
            LOG_ERROR("shmat failed: {}", strerror(errno));
            return;
        }
        header_ = static_cast<CircularBufferHeader*>(shm_ptr);
    }
    is_initialized_ = true;
}
// 兼容旧接口（接收端用，不需要 stride）
void StitchCircularBuffer::initialize(const std::string& shm_name, int width, int height, bool create_new) {
    // 默认用 width 估算 stride，仅用于接收端 attach
    int y_stride = ((width + 31) / 32) * 32;
    int uv_stride = ((width + 31) / 32) * 32;
    initialize(shm_name, width, height, create_new, y_stride, uv_stride);
}

// 2. 入栈 - 写入do_stitch拼接结果到循环队列
void StitchCircularBuffer::push_stitch_image(AVFrame* stitched_frame) {
    if (!is_initialized_ || !stitched_frame || !header_) {
       // LOG_ERROR("Not initialized or invalid frame");
        return;
    }
    
    // 检查是否满了
    if (header_->count >= MAX_FRAMES) {
        // 满了，覆盖最老的帧 (循环覆盖)
        header_->frames_dropped++;
       // LOG_WARN("Buffer full, overwriting oldest frame (dropped: {})", header_->frames_dropped);
    }
    
    // 获取写入槽
    StitchFrame* frame_slot = get_frame_slot(header_->head);
    if (!frame_slot) {
        //LOG_ERROR("Failed to get frame slot");
        return;
    }
    
   // LOG_DEBUG("Pushing frame to slot {} (seq: {})", header_->head, header_->sequence_counter);
    
    // 标记写入中
    frame_slot->ready = 1; // 写入中
    
    // === 保存do_stitch的AVFrame完整信息 ===
    frame_slot->width = stitched_frame->width;
    frame_slot->height = stitched_frame->height;
    frame_slot->format = stitched_frame->format;  // AV_PIX_FMT_CUDA
    frame_slot->pts = stitched_frame->pts;
    frame_slot->frame_sequence = header_->sequence_counter++;
    
    // === 保存原始GPU指针信息 (仅作记录) ===
    frame_slot->original_gpu_y_ptr = reinterpret_cast<uint64_t>(stitched_frame->data[0]);
    frame_slot->original_gpu_uv_ptr = reinterpret_cast<uint64_t>(stitched_frame->data[1]);
    frame_slot->original_linesize_y = stitched_frame->linesize[0];
    frame_slot->original_linesize_uv = stitched_frame->linesize[1];
    frame_slot->write_timestamp = get_now_time();
    
    // === 将GPU图像数据转换并复制到共享内存 ===
    // 获取CPU帧（如果是GPU格式需要转换）
    AVFrame* cpu_frame = nullptr;
    AVFrame* process_frame = stitched_frame;
    
    if (stitched_frame->format == AV_PIX_FMT_CUDA) {
        cpu_frame = av_frame_alloc();
        if (!cpu_frame || av_hwframe_transfer_data(cpu_frame, stitched_frame, 0) < 0) {
            LOG_ERROR("Failed to transfer GPU frame to CPU");
            if (cpu_frame) av_frame_free(&cpu_frame);
            frame_slot->ready = 0;
            return;
        }
        process_frame = cpu_frame;
    }
    
    // 获取实际的步长
    int src_y_stride = process_frame->linesize[0];
    int src_uv_stride = process_frame->linesize[1];
    
    // 确保步长是32字节对齐的（用于OpenGL渲染）
    int dst_y_stride = ((src_y_stride + 31) / 32) * 32;
    int dst_uv_stride = ((src_uv_stride + 31) / 32) * 32;
    
    // 检查分配的缓冲区是否足够（如果不够，说明初始化时估计不足）
    size_t required_y_size = dst_y_stride * stitched_frame->height;
    size_t required_uv_size = dst_uv_stride * (stitched_frame->height / 2);
    size_t required_total = required_y_size + required_uv_size;
    
    if (required_total > frame_slot->image_data_size) {
      //  LOG_WARN("Frame data size exceeds allocated space. Required: {}, Allocated: {}", 
        //         required_total, frame_slot->image_data_size);
        if (cpu_frame) av_frame_free(&cpu_frame);
        frame_slot->ready = 0;
        return;
    }
    
    // 保存步长信息
    frame_slot->y_stride = dst_y_stride;
    frame_slot->uv_stride = dst_uv_stride;
    
    // 获取目标缓冲区指针
    uint8_t* y_dest = frame_slot->get_y_data();
    uint8_t* uv_dest = frame_slot->get_uv_data();
    
    // 使用memcpy直接复制整个平面（快速方法）
    // 如果源步长和目标步长相同，直接复制整个平面（最快）
    if (src_y_stride == dst_y_stride) {
        // 步长相同，直接复制整个Y平面（一次memcpy）
        memcpy(y_dest, process_frame->data[0], src_y_stride * stitched_frame->height);
    } else {
        // 步长不同，需要逐行复制并调整步长
        for (int y = 0; y < stitched_frame->height; ++y) {
            // 复制实际图像数据（width字节）
            memcpy(y_dest + y * dst_y_stride, 
                   process_frame->data[0] + y * src_y_stride, 
                   stitched_frame->width);
            // 如果有对齐填充，填充剩余部分为0
            if (dst_y_stride > stitched_frame->width) {
                memset(y_dest + y * dst_y_stride + stitched_frame->width, 0, 
                       dst_y_stride - stitched_frame->width);
            }
        }
    }
    
    // UV平面复制（同样优化）
    if (src_uv_stride == dst_uv_stride) {
        // 步长相同，直接复制整个UV平面（一次memcpy）
        memcpy(uv_dest, process_frame->data[1], src_uv_stride * (stitched_frame->height / 2));
    } else {
        // 步长不同，需要逐行复制并调整步长
        for (int y = 0; y < stitched_frame->height / 2; ++y) {
            // 复制实际图像数据（width字节）
            memcpy(uv_dest + y * dst_uv_stride, 
                   process_frame->data[1] + y * src_uv_stride, 
                   stitched_frame->width);
            // 如果有对齐填充，填充剩余部分为0
            if (dst_uv_stride > stitched_frame->width) {
                memset(uv_dest + y * dst_uv_stride + stitched_frame->width, 0, 
                       dst_uv_stride - stitched_frame->width);
            }
        }
    }
    
    bool success = true;
    if (cpu_frame) {
        av_frame_free(&cpu_frame);
    }
    
    if (success) {
        // 标记为可读
        frame_slot->ready = 2; // 可读取
        
        // 更新循环队列状态
        header_->head = next_index(header_->head);
        if (header_->count < MAX_FRAMES) {
            header_->count++;
        } else {
            // 满了，tail也要向前移动
            header_->tail = next_index(header_->tail);
        }
        header_->total_pushed++;
        
       
    } else {
        frame_slot->ready = 0; // 恢复空闲状态
    }
}

bool StitchCircularBuffer::acquire_frame() {
    if (!is_initialized_ || !header_) {
        has_current_data_ = false;
        current_frame_ = nullptr;
        return false;
    }

    if (has_current_data_) {
        return true;
    }

    if (header_->count == 0) {
        has_current_data_ = false;
        current_frame_ = nullptr;
        return false;
    }

    StitchFrame* frame_slot = get_frame_slot(header_->tail);
    if (!frame_slot || frame_slot->ready != 2) {
        has_current_data_ = false;
        current_frame_ = nullptr;
        return false;
    }

    
    current_frame_ = frame_slot;
    has_current_data_ = true;
    frame_slot->ready = 3; // 标记为读取中
    return true;
}

void StitchCircularBuffer::release_frame() {
    if (!is_initialized_ || !header_ || !has_current_data_ || !current_frame_) {
        return;
    }

    
    current_frame_->ready = 0;
    header_->tail = next_index(header_->tail);
    if (header_->count > 0) {
        header_->count--;
    }
    header_->total_popped++;

    current_frame_ = nullptr;
    has_current_data_ = false;
}

// ====== 状态查询函数实现 ======

bool StitchCircularBuffer::is_ready() const {
    return is_initialized_;
}

bool StitchCircularBuffer::has_data() const {
    return has_current_data_ && current_frame_ != nullptr;
}

StitchFrame* StitchCircularBuffer::get_current_data() {
    if (has_current_data_ && current_frame_) {
        return current_frame_;
    }
    return nullptr;
}

bool StitchCircularBuffer::is_full() const {
    return header_ ? (header_->count >= MAX_FRAMES) : false;
}

bool StitchCircularBuffer::is_empty() const {
    return header_ ? (header_->count == 0) : true;
}

int StitchCircularBuffer::get_count() const {
    return header_ ? header_->count : 0;
}

void StitchCircularBuffer::print_stats() const {
    if (!header_) return;
    
   
}

// 4. 清理
void StitchCircularBuffer::cleanup() {
    if (!is_initialized_) {
        return;
    }
    
   
    if (header_) {
        print_stats(); // 显示最终统计
        
        // 分离共享内存
        if (shmdt(header_) == -1) {
            //LOG_ERROR("shmdt failed: {}", strerror(errno));
        } else {
           // LOG_DEBUG("Detached from shared memory");
        }
        
        if (is_creator_ && shm_id_ != -1) {
            // 删除共享内存
            if (shmctl(shm_id_, IPC_RMID, nullptr) == -1) {
               // LOG_ERROR("shmctl IPC_RMID failed: {}", strerror(errno));
            } else {
               // LOG_INFO("Removed shared memory: {}", shm_name_);
            }
        }
    }
    
    header_ = nullptr;
    current_frame_ = nullptr;
    shm_id_ = -1;
    is_initialized_ = false;
    has_current_data_ = false;
    
   // LOG_INFO("Cleanup completed for: {}", shm_name_);
}