#pragma once
#include "safe_queue.hpp"
#include <memory>

// 二阶段需要修改的内容

class TaskManager; // 提前声明，防止头文件嵌套

enum ChannelType {
    P2C,
    C2P
};

class Channel {
public:
    virtual bool bind(TaskManager* producer, TaskManager* consumer, ChannelType type) = 0;
};

class FrameChannel : public Channel {
    std::shared_ptr<safe_queue<T_Frame>> m_data;
public:
    virtual bool bind(TaskManager* producer, TaskManager* consumer, ChannelType type);
};

class PacketChannel : public Channel {
    std::shared_ptr<safe_queue<T_Packet>> m_data;
public:
    virtual bool bind(TaskManager* producer, TaskManager* consumer, ChannelType type);
}