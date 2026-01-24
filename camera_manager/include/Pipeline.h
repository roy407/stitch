#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "CallbackConsumer.h"
#include "LogConsumer.h"

class DecoderConsumer;
class EncoderConsumer;
class StitchConsumer;
struct CameraConfig;
struct PipelineConfig;
class TaskManager;

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