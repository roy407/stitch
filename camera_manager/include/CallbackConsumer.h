#pragma once

#include "Consumer.h"
#include "Channel.h"

// 主要是为了统一分配线程，但是目前还没想好具体咋用
class CallbackConsumer : public Consumer {
protected:
    FrameChannel* m_channel{nullptr};
    void(*m_callback)(Frame);
public:
    void setChannel(FrameChannel* channel);
    void setCallback(void(*callback)(Frame));
    CallbackConsumer();
    virtual ~CallbackConsumer();
    virtual void start();
    virtual void stop();
    virtual void run();
};