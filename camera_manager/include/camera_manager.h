#pragma once
#include <vector>
#include "safe_queue.hpp"
#include "tools.hpp"
#include "Pipeline.h"

class camera_manager {
public:
    static camera_manager* GetInstance();
    void start();
    void stop();
    void initPipeline();
    FrameChannel* getStitchCameraStream(int pipeline_id) const; // 相机拼接图
    FrameChannel* getSingleCameraSubStream(int cam_id) const; // 单相机子码流，非拼接图
    size_t getResizeCameraStreamCount() const;
private:
    camera_manager();
    ~camera_manager();
    std::vector<Pipeline*> m_pipelines;
    LogConsumer* m_log;
    bool m_running{false};
};