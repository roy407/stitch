#pragma once 

#include <sys/shm.h>
#include <sys/ipc.h>
#include <sys/types.h>
#include <semaphore.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <string>
#include <stdexcept>

extern "C" {
    #include <libavutil/frame.h>
    #include <libavutil/hwcontext.h>
}

// 单帧拼接结果数据结构
struct StitchFrame {
    volatile int ready;          // 0=空闲, 1=写入中, 2=可读取
    
    // === do_stitch返回的AVFrame完整信息 ===
    int width;                   // 拼接图像宽度 (如9600)
    int height;                  // 拼接图像高度 (如1080)
    int format;                  // AV_PIX_FMT_CUDA
    int64_t pts;                 // 时间戳
    uint32_t frame_sequence;     // 帧序列号
    
    // === 原始GPU指针信息 (记录用，无法跨进程使用) ===
    uint64_t original_gpu_y_ptr;   // 原始GPU Y指针地址
    uint64_t original_gpu_uv_ptr;  // 原始GPU UV指针地址
    int original_linesize_y;       // GPU中Y平面步长
    int original_linesize_uv;      // GPU中UV平面步长
    
    // === 共享内存中的步长信息 (用于快速访问) ===
    int y_stride;                  // Y平面步长 (存储时使用)
    int uv_stride;                 // UV平面步长 (存储时使用)
    
    // === 拼接后的图像数据 (在共享内存中) ===
    size_t image_data_size;        // 图像总大小
    uint64_t write_timestamp;      // 写入时间戳
    
    // 获取拼接后图像的Y和UV指针 (相对于此结构体，基于步长)
    uint8_t* get_y_data() { 
        return reinterpret_cast<uint8_t*>(this + 1); 
    }
    uint8_t* get_uv_data() { 
        // 基于y_stride计算UV数据位置
        return get_y_data() + (y_stride * height); 
    }
};

// 循环缓冲区控制头部
struct CircularBufferHeader {
                  // 魔数 0xABCD1234
    int max_frames;              // 最大帧数 (10)
    size_t frame_slot_size;      // 单个帧槽大小
    
    // === 循环队列控制 ===
    volatile int head;           // 写入位置 (生产者)
    volatile int tail;           // 读取位置 (消费者)  
    volatile int count;          // 当前帧数
    
    // === 统计信息 ===
    uint64_t total_pushed;       // 总写入帧数
    uint64_t total_popped;       // 总读取帧数
    uint64_t frames_dropped;     // 丢弃帧数
    uint32_t sequence_counter;   // 序列计数器
    
    // 帧数据紧跟在此结构后面
    // 布局: [Header][Frame0+Data0][Frame1+Data1]...[Frame9+Data9]
};

class StitchCircularBuffer {
public:
    StitchCircularBuffer();
    ~StitchCircularBuffer();
    
    // ====== 四个核心void接口 ======
    
    // 1. 初始化 - 创建10帧循环缓冲区（生产端用，必须传实际 stride）
    void initialize(const std::string& shm_name, int width, int height, bool create_new, int y_stride, int uv_stride);
    // 兼容旧接口（接收端用，不需要 stride）
    void initialize(const std::string& shm_name, int width, int height, bool create_new = true);
    
    // 2. 入栈 - 写入do_stitch拼接结果到循环队列
    void push_stitch_image(AVFrame* stitched_frame);
    
    // 3. 获取最老的帧但不立即释放
    bool acquire_frame();
    
    // 4. 释放当前帧（原 pop 操作）
    void release_frame();
    
    // 4. 清理
    void cleanup();
    
    // ====== 状态查询接口 ======
    bool is_ready() const;                    // 是否初始化完成
    bool has_data() const;                    // 是否有可用数据
    StitchFrame* get_current_data();          // 获取当前数据指针
    bool is_full() const;                     // 是否已满
    bool is_empty() const;                    // 是否为空
    int get_count() const;                    // 当前缓存帧数
    void print_stats() const;                 // 打印统计信息
    
private:
    std::string shm_name_;
    int shm_id_;
    CircularBufferHeader* header_;            // 循环缓冲区头部
    StitchFrame* current_frame_;              // 当前读取的帧指针
    size_t shm_size_;
    bool is_creator_;
    bool is_initialized_;
    bool has_current_data_;
    
    // === 内部辅助方法 ===
    StitchFrame* get_frame_slot(int index);    // 获取指定索引的帧槽
    size_t calculate_frame_slot_size(int width, int height, int y_stride, int uv_stride);  // 计算帧槽大小（基于步长）
    int next_index(int current);               // 获取下一个索引 (循环)
};