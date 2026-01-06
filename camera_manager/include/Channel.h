#pragma once
#include "safe_queue.hpp"
#include <memory>

class PacketChannel {
    safe_queue<Packet> m_data;
public:
    bool recv(Packet& out) {
        return m_data.wait_and_pop(out);
    }
    void send(Packet& p) {
        m_data.push(p);
    }
    void clear() {
        m_data.clear();
    }
    void stop() {
        m_data.stop();
    }
};

class FrameChannel {
    safe_queue<Frame> m_data;
public:
    bool recv(Frame& out) {
        // LOG_DEBUG("m data size is {},before recv",m_data.size());
        return m_data.wait_and_pop(out);
    }
    void send(Frame& p) {
        // LOG_DEBUG("m data size is {},before send",m_data.size());
        m_data.push(p);
        // LOG_DEBUG("m data size is {},after send",m_data.size());
    }
    void clear() {
        // LOG_DEBUG("m data size is {},before clear",m_data.size());
        m_data.clear();
        // LOG_DEBUG("m data size is {},after clear",m_data.size());
    }
    void stop() {
        // LOG_DEBUG("m data size is {},before stop",m_data.size());
        m_data.stop();
        // LOG_DEBUG("m data size is {},after stop",m_data.size());
    }
    void show_size()
    {
        LOG_DEBUG("m data size is {}",m_data.size());
    }
};