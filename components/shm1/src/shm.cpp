#include "shm.h"
#include <iostream>
#include <cstring>
#include <errno.h>
extern "C" {
    #include <libavutil/hwcontext.h>
}

#define CIRCULAR_BUFFER_MAGIC 0xABCD1234
#define MAX_FRAMES 10

StitchCircularBuffer::StitchCircularBuffer() 
    : shm_id_(-1), header_(nullptr), current_frame_(nullptr), is_creator_(false), 
      is_initialized_(false), has_current_data_(false) {}

StitchCircularBuffer::~StitchCircularBuffer() {
    cleanup();
}

// 计算单个帧槽大小 (StitchFrame + Y数据 + UV数据)
size_t StitchCircularBuffer::calculate_frame_slot_size(int width, int height) {
    size_t frame_struct_size = sizeof(StitchFrame);
    size_t y_data_size = width * height;
    size_t uv_data_size = width * height / 2; // NV12格式
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

// 1. 初始化 - 创建10帧循环缓冲区
void StitchCircularBuffer::initialize(const std::string& shm_name, int width, int height, bool create_new) {
    if (is_initialized_) {
        std::cerr << "Already initialized" << std::endl;
        return;
    }
    
    shm_name_ = shm_name;
    
    // 计算共享内存总大小: 头部 + 10个帧槽
    size_t header_size = sizeof(CircularBufferHeader);
    size_t frame_slot_size = calculate_frame_slot_size(width, height);
    shm_size_ = header_size + (frame_slot_size * MAX_FRAMES);
    
    // 生成key
    key_t key = 0;
    for (char c : shm_name) {
        key = key * 31 + c;
    }
    
    if (create_new) {
        // 尝试删除已存在的共享内存
        int old_shm_id = shmget(key, 0, 0);
        if (old_shm_id != -1) {
            shmctl(old_shm_id, IPC_RMID, nullptr);
        }
        
        // 创建新的共享内存
        shm_id_ = shmget(key, shm_size_, IPC_CREAT | 0666);
        if (shm_id_ == -1) {
            perror("shmget create failed");
            return;
        }
        is_creator_ = true;
    } else {
        // 连接到现有共享内存
        shm_id_ = shmget(key, 0, 0);
        if (shm_id_ == -1) {
            perror("shmget attach failed");
            return;
        }
        is_creator_ = false;
    }
    
    // 连接到共享内存
    void* shm_ptr = shmat(shm_id_, nullptr, 0);
    if (shm_ptr == (void*)-1) {
        perror("shmat failed");
        return;
    }
    
    header_ = static_cast<CircularBufferHeader*>(shm_ptr);
    
    if (create_new) {
        // 初始化循环缓冲区头部
        memset(header_, 0, shm_size_);
        header_->magic = CIRCULAR_BUFFER_MAGIC;
        header_->max_frames = MAX_FRAMES;
        header_->frame_slot_size = frame_slot_size;
        header_->head = 0;
        header_->tail = 0;
        header_->count = 0;
        header_->total_pushed = 0;
        header_->total_popped = 0;
        header_->frames_dropped = 0;
        header_->sequence_counter = 1;
        
        // 初始化所有帧槽为空闲状态
        for (int i = 0; i < MAX_FRAMES; i++) {
            StitchFrame* frame = get_frame_slot(i);
            if (frame) {
                frame->ready = 0; // 空闲
                frame->width = width;
                frame->height = height;
                frame->image_data_size = width * height * 3 / 2; // NV12
            }
        }
        
        std::cout << "Created circular buffer: " << shm_name 
                 << " (" << shm_size_ << " bytes, " << MAX_FRAMES << " frames, "
                 << frame_slot_size << " bytes per frame)" << std::endl;
    } else {
        // 验证魔数
        if (header_->magic != CIRCULAR_BUFFER_MAGIC) {
            std::cerr << " Invalid shared memory magic number" << std::endl;
            return;
        }
        std::cout << " Attached to circular buffer: " << shm_name << std::endl;
    }
    
    is_initialized_ = true;
}

// 2. 入栈 - 写入do_stitch拼接结果到循环队列
void StitchCircularBuffer::push_stitch_image(AVFrame* stitched_frame) {
    if (!is_initialized_ || !stitched_frame || !header_) {
        std::cerr << " Not initialized or invalid frame" << std::endl;
        return;
    }
    
    // 检查是否满了
    if (header_->count >= MAX_FRAMES) {
        // 满了，覆盖最老的帧 (循环覆盖)
        header_->frames_dropped++;
        std::cout << " Buffer full, overwriting oldest frame (dropped: " 
                 << header_->frames_dropped << ")" << std::endl;
    }
    
    // 获取写入槽
    StitchFrame* frame_slot = get_frame_slot(header_->head);
    if (!frame_slot) {
        std::cerr << "Failed to get frame slot" << std::endl;
        return;
    }
    
    std::cout << " Pushing frame to slot " << header_->head 
             << " (seq: " << header_->sequence_counter << ")" << std::endl;
    
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
    uint8_t* y_dest = frame_slot->get_y_data();
    uint8_t* uv_dest = frame_slot->get_uv_data();
    
    bool success = false;
    
    if (stitched_frame->format == AV_PIX_FMT_CUDA) {
        // GPU转CPU
        AVFrame* cpu_frame = av_frame_alloc();
        if (cpu_frame && av_hwframe_transfer_data(cpu_frame, stitched_frame, 0) == 0) {
            // 复制Y平面
            for (int y = 0; y < stitched_frame->height; ++y) {
                memcpy(y_dest + y * stitched_frame->width,
                       cpu_frame->data[0] + y * cpu_frame->linesize[0],
                       stitched_frame->width);
            }
            
            // 复制UV平面
            for (int y = 0; y < stitched_frame->height / 2; ++y) {
                memcpy(uv_dest + y * stitched_frame->width,
                       cpu_frame->data[1] + y * cpu_frame->linesize[1],
                       stitched_frame->width);
            }
            
            success = true;
        } else {
            std::cerr << "❌ Failed to transfer GPU frame to CPU" << std::endl;
        }
        if (cpu_frame) av_frame_free(&cpu_frame);
    } else if (stitched_frame->format == AV_PIX_FMT_NV12) {
        // 直接复制CPU格式数据
        for (int y = 0; y < stitched_frame->height; ++y) {
            memcpy(y_dest + y * stitched_frame->width,
                   stitched_frame->data[0] + y * stitched_frame->linesize[0],
                   stitched_frame->width);
        }
        for (int y = 0; y < stitched_frame->height / 2; ++y) {
            memcpy(uv_dest + y * stitched_frame->width,
                   stitched_frame->data[1] + y * stitched_frame->linesize[1],
                   stitched_frame->width);
        }
        success = true;
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
        
        std::cout << "✅ Frame pushed successfully (count: " << header_->count << "/10)" << std::endl;
    } else {
        frame_slot->ready = 0; // 恢复空闲状态
        std::cerr << "❌ Failed to push frame" << std::endl;
    }
}

// 3. 出栈 - 从循环队列读取最老的帧 (FIFO)
void StitchCircularBuffer::pop_stitch_image() {
    if (!is_initialized_ || !header_) {
        std::cerr << "❌ Not initialized" << std::endl;
        has_current_data_ = false;
        current_frame_ = nullptr;
        return;
    }
    
    // 检查是否为空
    if (header_->count == 0) {
        has_current_data_ = false;
        current_frame_ = nullptr;
        return;
    }
    
    // 获取读取槽 (最老的帧)
    StitchFrame* frame_slot = get_frame_slot(header_->tail);
    if (!frame_slot || frame_slot->ready != 2) {
        has_current_data_ = false;
        current_frame_ = nullptr;
        return;
    }
    
    std::cout << " Popping frame from slot " << header_->tail 
             << " (seq: " << frame_slot->frame_sequence << ")" << std::endl;
    
    // 设置当前帧指针
    current_frame_ = frame_slot;
    has_current_data_ = true;
    
    // 更新循环队列状态 (FIFO)
    header_->tail = next_index(header_->tail);
    header_->count--;
    header_->total_popped++;
    
    // 标记该槽为空闲 (可以被覆盖)
    frame_slot->ready = 0;
    
    std::cout << " Frame popped successfully (remaining: " << header_->count << "/10)" << std::endl;
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
    
    std::cout << "=== 循环缓冲区统计 ===" << std::endl;
    std::cout << "最大帧数: " << header_->max_frames << std::endl;
    std::cout << "当前帧数: " << header_->count << std::endl;
    std::cout << "Head: " << header_->head << ", Tail: " << header_->tail << std::endl;
    std::cout << "总写入: " << header_->total_pushed << std::endl;
    std::cout << "总读取: " << header_->total_popped << std::endl;
    std::cout << "丢弃帧数: " << header_->frames_dropped << std::endl;
    std::cout << "帧槽大小: " << header_->frame_slot_size << " bytes" << std::endl;
    std::cout << "序列计数: " << header_->sequence_counter << std::endl;
    std::cout << "========================" << std::endl;
}

// 4. 清理
void StitchCircularBuffer::cleanup() {
    if (!is_initialized_) {
        return;
    }
    
    std::cout << "🧹 Cleaning up circular buffer..." << std::endl;
    
    if (header_) {
        print_stats(); // 显示最终统计
        
        // 分离共享内存
        if (shmdt(header_) == -1) {
            perror("shmdt failed");
        } else {
            std::cout << "✅ Detached from shared memory" << std::endl;
        }
        
        if (is_creator_ && shm_id_ != -1) {
            // 删除共享内存
            if (shmctl(shm_id_, IPC_RMID, nullptr) == -1) {
                perror("shmctl IPC_RMID failed");
            } else {
                std::cout << "✅ Removed shared memory: " << shm_name_ << std::endl;
            }
        }
    }
    
    header_ = nullptr;
    current_frame_ = nullptr;
    shm_id_ = -1;
    is_initialized_ = false;
    has_current_data_ = false;
    
    std::cout << "✅ Cleanup completed for: " << shm_name_ << std::endl;
}