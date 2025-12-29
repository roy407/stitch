// YUVFrameReader.cpp
#include "YUVFrameReader.h"
#include <cstring>
#include <vector>
#include <algorithm>
extern "C" {
    #include <libavutil/imgutils.h>
    #include <libavutil/hwcontext.h>
    #include <libavutil/hwcontext_cuda.h>
    #include <libavutil/cpu.h>
}
#include "cuda_handle_init.h"

YUVFrameReader::YUVFrameReader(int cam_id, const std::string& yuv_path, int w, int h, AVPixelFormat fmt) 
    : cam_id(cam_id), yuv_file_path(yuv_path), width(w), height(h), pix_fmt(fmt) {
    m_name = "YUVFrameReader_" + std::to_string(cam_id);
    m_channel2stitch = new FrameChannel;
    m_channel2resize = new FrameChannel;
    
    // 初始化CUDA上下文
    initCudaContext();
}

YUVFrameReader::~YUVFrameReader() {
    delete m_channel2stitch;
    delete m_channel2resize;
    
    // 清理CPU缓存
    cpu_buffer.clear();
    cpu_buffer.shrink_to_fit();
    
    // 清理CUDA流
    if (cuda_stream) {
        cudaStreamDestroy(cuda_stream);
        cuda_stream = nullptr;
    }
    
    // 清理硬件帧上下文
    if (hw_frames_ctx) {
        av_buffer_unref(&hw_frames_ctx);
        hw_frames_ctx = nullptr;
    }
}

bool YUVFrameReader::initCudaContext() {
    int frame_size = getFrameSize();
    if (frame_size <= 0) {
        LOG_ERROR("Invalid frame size for camera {}", cam_id);
        return false;
    }
    
    // 分配CPU缓存
    cpu_buffer.resize(frame_size);
    
    // 创建CUDA流
    cudaError_t err = cudaStreamCreate(&cuda_stream);
    if (err != cudaSuccess) {
        LOG_ERROR("Failed to create CUDA stream for camera {}: {}", 
                 cam_id, cudaGetErrorString(err));
        return false;
    }
    
    // 创建硬件帧上下文
    AVBufferRef* device_ctx = cuda_handle_init::GetGPUDeviceHandle();
    if (!device_ctx) {
        LOG_WARN("No CUDA device available, will use CPU fallback for camera {}", cam_id);
        use_cuda = false;
        return true; // 回退到CPU模式
    }
    
    hw_frames_ctx = av_hwframe_ctx_alloc(device_ctx);
    if (!hw_frames_ctx) {
        LOG_WARN("Failed to allocate hardware frame context for camera {}, using CPU fallback", 
                cam_id);
        use_cuda = false;
        return true;
    }
    
    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)hw_frames_ctx->data;
    frames_ctx->format = AV_PIX_FMT_CUDA;
    
    // 设置软件格式
    if (pix_fmt == AV_PIX_FMT_YUV420P || pix_fmt == AV_PIX_FMT_YUVJ420P) {
        frames_ctx->sw_format = AV_PIX_FMT_YUV420P;
    } else if (pix_fmt == AV_PIX_FMT_NV12) {
        frames_ctx->sw_format = AV_PIX_FMT_NV12;
    } else if (pix_fmt == AV_PIX_FMT_NV21) {
        frames_ctx->sw_format = AV_PIX_FMT_NV21;
    } else {
        LOG_WARN("Unsupported pixel format {} for GPU, using CPU fallback", 
                av_get_pix_fmt_name(pix_fmt));
        av_buffer_unref(&hw_frames_ctx);
        hw_frames_ctx = nullptr;
        use_cuda = false;
        return true;
    }
    
    frames_ctx->width = width;
    frames_ctx->height = height;
    frames_ctx->initial_pool_size = 3; // 3个缓冲：当前帧、stitch帧、resize帧
    
    int ret = av_hwframe_ctx_init(hw_frames_ctx);
    if (ret < 0) {
        LOG_WARN("Failed to initialize hardware frame context for camera {}: {}, using CPU fallback", 
                cam_id, ret);
        av_buffer_unref(&hw_frames_ctx);
        hw_frames_ctx = nullptr;
        use_cuda = false;
        return true;
    }
    
    LOG_INFO("GPU context initialized for camera {}", cam_id);
    use_cuda = true;
    return true;
}

int YUVFrameReader::getFrameSize() const {
    int y_size = width * height;
    
    switch (pix_fmt) {
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_YUVJ420P:
            return y_size * 3 / 2; // Y + U + V
            
        case AV_PIX_FMT_NV12:
        case AV_PIX_FMT_NV21:
            return y_size * 3 / 2; // Y + UV
            
        default:
            LOG_WARN("Unsupported pixel format: {}, defaulting to YUV420P size", 
                    av_get_pix_fmt_name(pix_fmt));
            return y_size * 3 / 2;
    }
}

AVFrame* YUVFrameReader::createGPUFrame() {
    if (!use_cuda || !hw_frames_ctx) {
        LOG_ERROR("GPU not available or not initialized for camera {}", cam_id);
        return nullptr;
    }
    
    AVFrame* gpu_frame = av_frame_alloc();
    if (!gpu_frame) {
        LOG_ERROR("Failed to allocate GPU frame for camera {}", cam_id);
        return nullptr;
    }
    
    gpu_frame->format = AV_PIX_FMT_CUDA;
    gpu_frame->width = width;
    gpu_frame->height = height;
    
    // 从硬件帧上下文获取缓冲区
    int ret = av_hwframe_get_buffer(hw_frames_ctx, gpu_frame, 0);
    if (ret < 0) {
        LOG_ERROR("Failed to get GPU buffer for camera {}: {}", cam_id, ret);
        av_frame_free(&gpu_frame);
        return nullptr;
    }
    
    return gpu_frame;
}

AVFrame* YUVFrameReader::createCPUFrame() {
    AVFrame* cpu_frame = av_frame_alloc();
    if (!cpu_frame) {
        LOG_ERROR("Failed to allocate CPU frame for camera {}", cam_id);
        return nullptr;
    }
    
    cpu_frame->format = pix_fmt;
    cpu_frame->width = width;
    cpu_frame->height = height;
    
    // 分配CPU内存
    int ret = av_frame_get_buffer(cpu_frame, 32); // 32字节对齐
    if (ret < 0) {
        LOG_ERROR("Failed to allocate CPU buffer for camera {}: {}", cam_id, ret);
        av_frame_free(&cpu_frame);
        return nullptr;
    }
    
    return cpu_frame;
}

bool YUVFrameReader::fillCPUFrameFromBuffer(AVFrame* cpu_frame, const uint8_t* buffer) {
    if (!cpu_frame || !buffer) {
        LOG_ERROR("Invalid parameters for fillCPUFrameFromBuffer");
        return false;
    }
    
    int y_size = width * height;
    
    if (pix_fmt == AV_PIX_FMT_YUV420P || pix_fmt == AV_PIX_FMT_YUVJ420P) {
        // YUV420P: Y + U + V 三个平面
        const uint8_t* y_plane = buffer;
        const uint8_t* u_plane = buffer + y_size;
        const uint8_t* v_plane = buffer + y_size + (y_size / 4);
        
        // 复制Y平面
        for (int i = 0; i < height; i++) {
            memcpy(cpu_frame->data[0] + i * cpu_frame->linesize[0],
                   y_plane + i * width,
                   width);
        }
        
        // 复制U平面
        for (int i = 0; i < height / 2; i++) {
            memcpy(cpu_frame->data[1] + i * cpu_frame->linesize[1],
                   u_plane + i * (width / 2),
                   width / 2);
        }
        
        // 复制V平面
        for (int i = 0; i < height / 2; i++) {
            memcpy(cpu_frame->data[2] + i * cpu_frame->linesize[2],
                   v_plane + i * (width / 2),
                   width / 2);
        }
        
    } else if (pix_fmt == AV_PIX_FMT_NV12) {
        // NV12: Y平面 + UV交错平面
        const uint8_t* y_plane = buffer;
        const uint8_t* uv_plane = buffer + y_size;
        
        // 复制Y平面
        for (int i = 0; i < height; i++) {
            memcpy(cpu_frame->data[0] + i * cpu_frame->linesize[0],
                   y_plane + i * width,
                   width);
        }
        
        // 复制UV交错平面
        for (int i = 0; i < height / 2; i++) {
            memcpy(cpu_frame->data[1] + i * cpu_frame->linesize[1],
                   uv_plane + i * width,
                   width);
        }
        
    } else if (pix_fmt == AV_PIX_FMT_NV21) {
        // NV21: Y平面 + VU交错平面
        const uint8_t* y_plane = buffer;
        const uint8_t* vu_plane = buffer + y_size;
        
        // 复制Y平面
        for (int i = 0; i < height; i++) {
            memcpy(cpu_frame->data[0] + i * cpu_frame->linesize[0],
                   y_plane + i * width,
                   width);
        }
        
        // 复制VU交错平面
        for (int i = 0; i < height / 2; i++) {
            memcpy(cpu_frame->data[1] + i * cpu_frame->linesize[1],
                   vu_plane + i * width,
                   width);
        }
        
    } else {
        LOG_ERROR("Unsupported pixel format in fillCPUFrameFromBuffer: {}", 
                 av_get_pix_fmt_name(pix_fmt));
        return false;
    }
    
    return true;
}

AVFrame* YUVFrameReader::transferToGPU(AVFrame* cpu_frame) {
    if (!use_cuda || !cpu_frame) {
        return nullptr;
    }
    
    // 创建GPU帧
    AVFrame* gpu_frame = createGPUFrame();
    if (!gpu_frame) {
        return nullptr;
    }
    
    // 使用FFmpeg的transfer API将CPU数据复制到GPU
    int ret = av_hwframe_transfer_data(gpu_frame, cpu_frame, 0);
    if (ret < 0) {
        LOG_ERROR("Failed to transfer frame to GPU for camera {}: {}", cam_id, ret);
        av_frame_free(&gpu_frame);
        return nullptr;
    }
    
    // 同步CUDA流
    if (cuda_stream) {
        cudaStreamSynchronize(cuda_stream);
    }
    
    return gpu_frame;
}

bool YUVFrameReader::readFrameData() {
    int frame_size = getFrameSize();
    
    // 从文件读取数据到CPU缓冲区
    file_stream.read(reinterpret_cast<char*>(cpu_buffer.data()), frame_size);
    size_t bytes_read = file_stream.gcount();
    
    if (bytes_read != frame_size) {
        if (file_stream.eof()) {
            // 文件结束，重新开始
            file_stream.clear();
            file_stream.seekg(0, std::ios::beg);
            
            LOG_INFO("YUV file ended, restarting from beginning for camera {}", cam_id);
            
            // 重新读取
            file_stream.read(reinterpret_cast<char*>(cpu_buffer.data()), frame_size);
            bytes_read = file_stream.gcount();
            
            if (bytes_read != frame_size) {
                LOG_ERROR("Failed to restart reading YUV file for camera {}", cam_id);
                return false;
            }
            
            // 重置帧计数
            frame_count = 0;
        } else {
            LOG_ERROR("Failed to read complete frame. Expected {} bytes, got {} bytes", 
                     frame_size, bytes_read);
            return false;
        }
    }
    
    return true;
}

void YUVFrameReader::start() {
    file_stream.open(yuv_file_path, std::ios::binary);
    if (!file_stream.is_open()) {
        LOG_ERROR("Cannot open YUV file: {}", yuv_file_path);
        return;
    }
    
    // 输出文件信息
    file_stream.seekg(0, std::ios::end);
    size_t file_size = file_stream.tellg();
    file_stream.seekg(0, std::ios::beg);
    
    int frame_size = getFrameSize();
    size_t total_frames = file_size / frame_size;
    
    LOG_INFO("YUV file opened: {}", yuv_file_path);
    LOG_INFO("  File size: {} bytes", file_size);
    LOG_INFO("  Frame size: {} bytes", frame_size);
    LOG_INFO("  Estimated frames: {}", total_frames);
    LOG_INFO("  Pixel format: {}", av_get_pix_fmt_name(pix_fmt));
    LOG_INFO("  GPU acceleration: {}", use_cuda ? "Enabled" : "Disabled (CPU fallback)");
    
    // 预分配CPU帧
    cpu_cached_frame = createCPUFrame();
    if (!cpu_cached_frame) {
        LOG_ERROR("Failed to create cached CPU frame for camera {}", cam_id);
        return;
    }
    
    TaskManager::start();
}

void YUVFrameReader::stop() {
    TaskManager::stop();
    
    if (file_stream.is_open()) {
        file_stream.close();
    }
    
    // 清理缓存的帧
    if (cpu_cached_frame) {
        av_frame_free(&cpu_cached_frame);
    }
    
    LOG_INFO("YUVFrameReader for camera {} stopped", cam_id);
}

// void YUVFrameReader::run() {
//     auto frame_interval = std::chrono::duration<double>(1.0 / fps);
//     // 添加帧率测量
//     auto last_measure_time = std::chrono::steady_clock::now();
//     int frames_in_second = 0;
//     while (running && file_stream) {
//         auto start_time = std::chrono::steady_clock::now();    

//         // 1. 读取帧数据
//         if (!readFrameData()) {
//             LOG_ERROR("Failed to read frame data for camera {}", cam_id);
//             break;
//         }
        
//         // 2. 填充到CPU帧
//         if (!fillCPUFrameFromBuffer(cpu_cached_frame, cpu_buffer.data())) {
//             LOG_ERROR("Failed to fill CPU frame for camera {}", cam_id);
//             break;
//         }
        
//         // 3. 设置时间戳
//         cpu_cached_frame->pts = frame_count;
//         cpu_cached_frame->pkt_dts = frame_count;
        
//         // 4. 创建要发送的帧
//         AVFrame* output_frame = nullptr;
        
//         if (use_cuda) {
//             // GPU路径：传输到GPU
//             output_frame = transferToGPU(cpu_cached_frame);
//             if (!output_frame) {
//                 LOG_WARN("GPU transfer failed, falling back to CPU for camera {}", cam_id);
//                 // 回退：直接使用CPU帧（需要复制）
//                 output_frame = av_frame_clone(cpu_cached_frame);
//             }
//         } else {
//             // CPU路径：直接复制CPU帧
//             output_frame = av_frame_clone(cpu_cached_frame);
//         }
        
//         if (!output_frame) {
//             LOG_ERROR("Failed to create output frame for camera {}", cam_id);
//             break;
//         }
        
//         // 5. 创建Frame对象并发送
//         Frame stitch_frame;
//         stitch_frame.cam_id = cam_id;
//         stitch_frame.m_data = output_frame;
//         stitch_frame.m_timestamp = get_now_time();
//         stitch_frame.m_costTimes.image_frame_cnt[cam_id] = ++frame_count;
//         stitch_frame.m_costTimes.when_get_decoded_frame[cam_id] = get_now_time();
        
//         try {
//             m_channel2stitch->send(stitch_frame);
//         } catch (const std::exception& e) {
//             LOG_WARN("Failed to send frame to stitch channel for camera {}: {}", 
//                     cam_id, e.what());
//             av_frame_free(&output_frame);
//         }
        
//         // 测量实际帧率
//         frames_in_second++;
//         auto now = std::chrono::steady_clock::now();
//         auto elapsed_since_measure = now - last_measure_time;
        
//         if (std::chrono::duration_cast<std::chrono::seconds>(elapsed_since_measure).count() >= 1) {
//             double actual_fps = frames_in_second / 
//                               std::chrono::duration<double>(elapsed_since_measure).count();
//             LOG_INFO("Camera {} - Set FPS: {:.1f}, Actual FPS: {:.1f}, Processing time: {:.1f}ms", 
//                     cam_id, fps, actual_fps,
//                     std::chrono::duration<double, std::milli>(now - start_time).count());
            
//             frames_in_second = 0;
//             last_measure_time = now;
//         }

//         // 6. 控制帧率
//         auto elapsed = std::chrono::steady_clock::now() - start_time;
//         if (elapsed < frame_interval) {
//             std::this_thread::sleep_for(frame_interval - elapsed);
//         }
        
//         // 7. 定期输出日志
//         if (frame_count % 100 == 0) {
//             LOG_DEBUG("Camera {} processed {} frames", cam_id, frame_count);
//         }
//     }
    
//     if (running) {
//         LOG_INFO("YUVFrameReader for camera {} finished processing", cam_id);
//     }
// }

void YUVFrameReader::run() {
    auto frame_interval = std::chrono::duration<double>(1.0 / fps);
    
    // 详细性能统计
    struct {
        double read_time{0};
        double fill_time{0};
        double transfer_time{0};
        double send_time{0};
        double total_time{0};
    } stats;
    int stat_count{0};
    int first_run=0;
    while (running && file_stream) {
        if(first_run==0)
        {
            auto total_start = std::chrono::steady_clock::now();
            
            // 1. 读取
            auto read_start = std::chrono::steady_clock::now();
            if (!readFrameData()) {
                LOG_ERROR("Failed to read frame data for camera {}", cam_id);
                break;
            }
            stats.read_time += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - read_start).count();
            
            // 2. 填充CPU帧
            auto fill_start = std::chrono::steady_clock::now();
            if (!fillCPUFrameFromBuffer(cpu_cached_frame, cpu_buffer.data())) {
                LOG_ERROR("Failed to fill CPU frame for camera {}", cam_id);
                break;
            }
            stats.fill_time += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - fill_start).count();
            
            // 3. 传输到GPU
            auto transfer_start = std::chrono::steady_clock::now();
            AVFrame* output_frame = nullptr;
            
            if (use_cuda) {
                output_frame = transferToGPU(cpu_cached_frame);
                if (!output_frame) {
                    LOG_WARN("GPU transfer failed, falling back to CPU for camera {}", cam_id);
                    output_frame = av_frame_clone(cpu_cached_frame);
                }
            } else {
                output_frame = av_frame_clone(cpu_cached_frame);
            }
            stats.transfer_time += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - transfer_start).count();
            
            if (!output_frame) {
                LOG_ERROR("Failed to create output frame for camera {}", cam_id);
                break;
            }
            
            // 4. 发送
            auto send_start = std::chrono::steady_clock::now();
            Frame stitch_frame;
            stitch_frame.cam_id = cam_id;
            stitch_frame.m_data = output_frame;
            stitch_frame.m_timestamp = get_now_time();
            stitch_frame.m_costTimes.image_frame_cnt[cam_id] = ++frame_count;
            stitch_frame.m_costTimes.when_get_decoded_frame[cam_id] = get_now_time();
            
            try {
                m_channel2stitch->send(stitch_frame);
            } catch (const std::exception& e) {
                LOG_WARN("Failed to send frame to stitch channel for camera {}: {}", 
                        cam_id, e.what());
                av_frame_free(&output_frame);
            }
            stats.send_time += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - send_start).count();
            
            stats.total_time += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - total_start).count();
            stat_count++;
            
            // 输出性能统计
            if (stat_count >= 30) {
                LOG_INFO("Camera {} performance breakdown (avg ms):", cam_id);
                LOG_INFO("  Read: {:.2f}", stats.read_time / stat_count);
                LOG_INFO("  Fill: {:.2f}", stats.fill_time / stat_count);
                LOG_INFO("  Transfer: {:.2f}", stats.transfer_time / stat_count);
                LOG_INFO("  Send: {:.2f}", stats.send_time / stat_count);
                LOG_INFO("  Total: {:.2f} (Max FPS: {:.1f})", 
                        stats.total_time / stat_count,
                        1000.0 / (stats.total_time / stat_count));
                
                // 重置统计
                memset(&stats, 0, sizeof(stats));
                stat_count = 0;
            }
            
            // 帧率控制
            auto elapsed = std::chrono::steady_clock::now() - total_start;
            if (elapsed < frame_interval) {
                std::this_thread::sleep_for(frame_interval - elapsed);
            }
            first_run=1;
            LOG_INFO("first is :{}",first_run);
        }
        else
            std::this_thread::sleep_for(std::chrono::duration<double>(0.01));
    }
}

// 辅助函数：验证YUV文件格式
bool YUVFrameReader::verifyFileFormat() {
    if (!file_stream.is_open()) {
        return false;
    }
    
    int frame_size = getFrameSize();
    if (frame_size <= 0) {
        return false;
    }
    
    // 读取第一帧
    file_stream.read(reinterpret_cast<char*>(cpu_buffer.data()), frame_size);
    size_t bytes_read = file_stream.gcount();
    file_stream.seekg(0, std::ios::beg);
    
    if (bytes_read != frame_size) {
        LOG_ERROR("File size mismatch. Expected {} bytes, got {}", frame_size, bytes_read);
        return false;
    }
    
    // 简单的YUV格式验证（检查Y分量范围）
    bool valid = true;
    int y_size = width * height;
    
    // 检查Y分量是否在合理范围内（可选）
    if (pix_fmt == AV_PIX_FMT_YUV420P || pix_fmt == AV_PIX_FMT_YUVJ420P ||
        pix_fmt == AV_PIX_FMT_NV12 || pix_fmt == AV_PIX_FMT_NV21) {
        // Y分量应该在0-255范围内（实际上可能超出）
        for (int i = 0; i < std::min(100, y_size); i++) {
            uint8_t y = cpu_buffer[i];
            if (y < 16 && y > 235) {
                // YUV标准范围是16-235，但可能有扩展范围
                // 这里只是简单检查，不严格限制
            }
        }
    }
    
    LOG_INFO("File format verified for camera {}", cam_id);
    return valid;
}

// 调试函数：保存一帧到文件用于验证
void YUVFrameReader::saveFrameForDebug(const std::string& filename) {
    if (cpu_buffer.empty()) {
        LOG_ERROR("No frame data available for debugging");
        return;
    }
    
    int frame_size = getFrameSize();
    std::ofstream debug_file(filename, std::ios::binary);
    
    if (debug_file.is_open()) {
        debug_file.write(reinterpret_cast<const char*>(cpu_buffer.data()), frame_size);
        debug_file.close();
        LOG_INFO("Saved debug frame to: {}", filename);
    }
}