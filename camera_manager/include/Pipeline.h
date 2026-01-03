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
#include "LogConsumer.h"
#include "StitchImpl.h"
#include "CallbackConsumer.h"

class Pipeline {
private:
    StitchConsumer* getStitchConsumer(int pipeline_id, std::string kernelTag);
    FrameChannel* initCameraProcessingFlows(const CameraConfig &cam);
    std::unordered_map<int, std::function<void(Callback_Handle)>> m_setCameraCallback; // cam_id -> CameraCallback;
    std::function<void(Callback_Handle)> m_setStitchCallback = nullptr;
    std::vector<TaskManager*> m_producerTask;
    std::vector<TaskManager*> m_consumerTask;
    static LogConsumer* m_log;
public:
    Pipeline(int pipeline_id);
    Pipeline(const PipelineConfig& p);
    ~Pipeline();
    static void setLogConsumer(LogConsumer* log);
    void start();
    void stop();
    bool setStitchStreamCallBack(Callback_Handle handle);
    void setCameraStreamCallBack(int cam_id, Callback_Handle handle);
    bool findCameraById(int cam_id);
    size_t getCameraStreamCount() const ;
};