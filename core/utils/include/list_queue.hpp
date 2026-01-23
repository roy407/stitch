#pragma once
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>
#include "tools.hpp"

template<typename T>
class list_queue {
public:
    explicit list_queue(size_t capacity = 10): 
        buffer(capacity), 
        capacity_(capacity),
        head(0), 
        tail(0), 
        count(0)  // 使用简单计数
    {}
    
    list_queue(const list_queue&) = delete;
    list_queue& operator=(const list_queue&) = delete;
 
    void push(const T& value);  // 改为void，总是成功
    bool try_pop(T& result);
    bool wait_and_pop(T& result);
    bool wait_and_front(T& result);
    
    bool empty() const;
    bool full() const;
    int size() const;
    int capacity() const;
    
    void clear();
    void stop();
    
    // 添加统计信息
    int frames{0};
    int packets{0};
    int frame_lost{0};
    int packet_lost{0};
    
private:
    void advance(size_t& index);
    void pop_and_free();
    
    mutable std::mutex mtx_;
    std::condition_variable cv;
    std::atomic_bool isStop{false};
    
    std::vector<T> buffer;
    size_t capacity_;
    size_t head;
    size_t tail;
    size_t count;  // 当前元素数量
};

template<typename T>
void list_queue<T>::advance(size_t& index) {
    index = (index + 1) % capacity_;
}

template<typename T>
void list_queue<T>::push(const T& value) {
    std::lock_guard<std::mutex> lock(mtx_);
    
    // 统计
    if constexpr (std::is_same<T, Packet>::value) {
        packets++;
    } else if constexpr (std::is_same<T, Frame>::value) {
        frames++;
    }
    
    // 如果满了，丢弃最旧的
    if (count == capacity_) {
        if constexpr (std::is_same<T, Packet>::value) {
            packet_lost++;
        } else if constexpr (std::is_same<T, Frame>::value) {
            frame_lost++;
        }
        // 释放最旧数据的资源
        if constexpr (std::is_same<T, Packet>::value) {
            Packet& pkt = buffer[head];
            av_packet_free(&pkt.m_data);
        } else if constexpr (std::is_same<T, Frame>::value) {
            Frame& frame = buffer[head];
            av_frame_free(&frame.m_data);
        }
        advance(head);  // 移动 head，丢弃最旧的
        // count 保持不变，因为会覆盖
    } else {
        count++;
    }
    
    // 放入新数据（覆盖 tail 位置）
    buffer[tail] = value;
    advance(tail);
    
    cv.notify_one();
}

template<typename T>
bool list_queue<T>::try_pop(T& result) {
    std::lock_guard<std::mutex> lock(mtx_);
    
    if (count == 0) {  // 使用count判断是否为空
        return false;
    }
    
    result = std::move(buffer[head]);
    advance(head);
    count--;
    
    return true;
}

template<typename T>
bool list_queue<T>::wait_and_pop(T& result) {
    std::unique_lock<std::mutex> lock(mtx_);
    
    cv.wait(lock, [this] { 
        return isStop.load() || count > 0;  // 使用count>0
    });
    
    if (isStop.load() && count == 0) {
        return false;
    }
    
    result = std::move(buffer[head]);
    advance(head);
    count--;
    
    return true;
}

template<typename T>
bool list_queue<T>::wait_and_front(T& result) {
    std::unique_lock<std::mutex> lock(mtx_);
    
    cv.wait(lock, [this] { 
        return isStop.load() || count > 0;  // 使用count>0
    });
    
    if (isStop.load() && count == 0) {
        return false;
    }
    
    result = buffer[head];  // 不移动，只查看
    return true;
}

template<typename T>
bool list_queue<T>::empty() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return count == 0;  // 使用count判断
}

template<typename T>
bool list_queue<T>::full() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return count == capacity_;  // 使用count判断
}

template<typename T>
int list_queue<T>::size() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return static_cast<int>(count);  // 直接返回count
}

template<typename T>
int list_queue<T>::capacity() const {
    return static_cast<int>(capacity_);
}

template<typename T>
void list_queue<T>::pop_and_free() {
    if (count == 0) return;
    
    if constexpr (std::is_same<T, Packet>::value) {
        Packet& pkt = buffer[head];
        av_packet_free(&pkt.m_data);
    } else if constexpr (std::is_same<T, Frame>::value) {
        Frame& frame = buffer[head];
        av_frame_free(&frame.m_data);
    }
    
    advance(head);
    count--;
}

template<typename T>
void list_queue<T>::clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    
    while (count > 0) {
        pop_and_free();
    }
}

template<typename T>
void list_queue<T>::stop() {
    isStop.store(true);
    cv.notify_all();    
}