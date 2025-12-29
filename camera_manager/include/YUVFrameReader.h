// YUVFrameReader.h
#pragma once
#include "TaskManager.h"
#include "Channel.h"
#include <fstream>

class YUVFrameReader : public TaskManager {
private:
    std::string yuv_file_path;
    int width, height;
    AVPixelFormat pix_fmt = AV_PIX_FMT_YUV420P;
    FrameChannel* m_channel2stitch = nullptr;
    FrameChannel* m_channel2resize = nullptr;
    std::ifstream file_stream;
    int64_t frame_count = 0;
    double fps = 1200.0;
    int cam_id;

public:
    YUVFrameReader(int cam_id, const std::string& yuv_path, int w, int h);
    ~YUVFrameReader();
    
    FrameChannel* getChannel2Stitch() { return m_channel2stitch; }
    FrameChannel* getChannel2Resize() { return m_channel2resize; }
    
    void start() override;
    void stop() override;
    void run() override;
};