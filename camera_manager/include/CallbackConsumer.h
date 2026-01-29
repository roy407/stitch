#pragma once

#include <functional>
#include <fstream>
#include <string>

#include "Channel.h"
#include "Consumer.h"

using Callback_Handle = std::function<void(Frame)>;

/**
 * @class CallbackConsumer
 * @brief 帧数据回调消费者类
 * 
 * 该类作为相机处理管道的终端消费者，通过回调机制向外部传输处理完成的帧数据。
 * 主要功能与特点：
 * 1. 提供标准化的回调接口，封装帧数据的线程安全传输
 * 2. 记录相机处理流水线中各级处理的时间戳，便于性能分析和优化
 * 3. 内部管理独立的消费者线程，外部使用者无需自行创建和管理线程
 * 4. 确保数据传递过程中的资源安全和异常处理
 * 
 * 设计目的：
 * - 简化外部接口：使用者只需关注数据处理逻辑，无需了解线程、同步等底层细节
 * - 提供性能可观测性：内建时间记录机制帮助分析处理延迟
 * - 保证安全性：确保回调调用期间相关资源的有效性和线程安全性
 * 
 * 使用示例：
 * @code
 * auto consumer = std::make_shared<CallbackConsumer>();
 * consumer->setChannel(frameChannel);
 * consumer->setCallback([](Frame frame) {
 *     // 处理接收到的帧数据
 *     processFrame(frame);
 * });
 * consumer->start();  // 内部自动创建并管理线程
 * @endcode
 */

class CallbackConsumer : public Consumer {
protected:
    FrameChannel* m_channel{nullptr};
    Callback_Handle m_callback;
    std::ofstream createFile();
    bool openTimingWatcher{false};
    std::string pipelineName;
public:
    void setChannel(FrameChannel* channel);
    void setCallback(Callback_Handle callback);
    CallbackConsumer();
    void setPipelineName(std::string name);
    void setTimingWatcher(bool enable);
    virtual ~CallbackConsumer();
    virtual void start();
    virtual void stop();
    virtual void run();
};