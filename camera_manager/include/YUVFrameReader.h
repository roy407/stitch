// YUVFrameReader.h
#ifndef YUV_FRAME_READER_H
#define YUV_FRAME_READER_H

#include <string>
#include <fstream>
#include <atomic>
#include <thread>
#include <chrono>
#include <memory>
#include "Channel.h"
#include "TaskManager.h"

extern "C" {
    #include <libavutil/frame.h>
    #include <libavutil/pixfmt.h>
    #include <libavutil/buffer.h>
}

class YUVFrameReader : public TaskManager {
public:
    YUVFrameReader(int cam_id, const std::string& yuv_path, int width, int height, 
                   AVPixelFormat pix_fmt = AV_PIX_FMT_YUV420P);
    ~YUVFrameReader();

    void setFPS(double frame_rate) { fps = frame_rate; }
    
    FrameChannel* getChannel2Stitch() { return m_channel2stitch; }
    FrameChannel* getChannel2Resize() { return m_channel2resize; }

    AVFrame* createDummyGPUFrames();
    bool initCudaContext();
    void start() override;
    void stop() override;
    
    FrameChannel* getStitchChannel() const { return m_channel2stitch; }
    FrameChannel* getResizeChannel() const { return m_channel2resize; }
    
    bool verifyFileFormat();
    void saveFrameForDebug(const std::string& filename);
    
private:
    void run() override;
    
    int getFrameSize() const;
    bool readFrameData();
    bool fillCPUFrameFromBuffer(AVFrame* cpu_frame, const uint8_t* buffer);
    AVFrame* createCPUFrame();
    AVFrame* createGPUFrame();
    AVFrame* transferToGPU(AVFrame* cpu_frame);
    
    // 成员变量
    int cam_id;
    std::string yuv_file_path;
    int width;
    int height;
    AVPixelFormat pix_fmt;
    
    std::ifstream file_stream;
    std::atomic<int> frame_count{0};
    
    FrameChannel* m_channel2stitch;
    FrameChannel* m_channel2resize;
    
    // GPU相关
    AVBufferRef* hw_frames_ctx{nullptr};
    cudaStream_t cuda_stream{nullptr};
    bool use_cuda{false};
    
    // CPU缓存
    std::vector<uint8_t> cpu_buffer;
    AVFrame* cpu_cached_frame{nullptr};
    
    // 帧率控制
    double fps{100.0};
};

#endif // YUV_FRAME_READER_H