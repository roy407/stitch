#pragma once
#pragma once
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <iostream>
#include <queue>
#include <chrono>

extern "C" {
    #include "libavformat/avformat.h"
    #include "libavcodec/avcodec.h"
    #include "libavutil/pixfmt.h" 
    #include "libavutil/pixdesc.h" 
    #include "libavutil/opt.h"
    #include "libavutil/log.h"
    #include "libavcodec/bsf.h"
}

#include "safe_queue.hpp"
#include "tools.hpp"
#include "DecoderConsumer.h"
#include "EncoderConsumer.h"
#include "AVFrameProducer.h"
#include "LogConsumer.h"
#include "StitchImpl.h"

class Pipeline {
private:
    StitchOps* getStitchOps(int pipeline_id, std::string Format, std::string kernelTag);
    std::unordered_map<int, FrameChannel*> m_resizeStream; // cam_id -> resize_stream
    std::vector<TaskManager*> m_producerTask;
    std::vector<TaskManager*> m_consumerTask;
    FrameChannel* m_stitchStream;
    static LogConsumer* m_log;
public:
    Pipeline(int pipeline_id);
    Pipeline(const PipelineConfig& p);
    static void setLogConsumer(LogConsumer* log);
    void start();
    void stop();
    FrameChannel* getStitchCameraStream() const;
    FrameChannel* getResizeCameraStream(int cam_id) const;
};