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
        return m_data.wait_and_pop(out);
    }
    void send(Frame& p) {
        m_data.push(p);
    }
    void clear() {
        m_data.clear();
    }
    void stop() {
        m_data.stop();
    }
};