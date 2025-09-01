#pragma once

/*---- C++模版必须全部写在头文件中！！！ -----*/
/*---- 如果要写在cpp文件中，那么只有预设实例可以使用 -----*/

#include <queue>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include "tools.hpp"

// 通过修改T_TEST是否被定义，来控制最终是否测试

#define T_TEST

#ifdef T_TEST
using T_Packet = std::pair<AVPacket*,costTimes>;
using T_Frame = std::pair<AVFrame*,costTimes>;
#else
using T_Packet = AVPacket*;
using T_Frame = AVFrame*;
#endif
template<typename T>
class safe_queue {
public:
    safe_queue() = default;
    safe_queue(const safe_queue&) = delete;
    safe_queue& operator=(const safe_queue&) = delete;

    void push(const T& value);
    bool try_pop(T& result);
    void wait_and_pop(T& result);
    void wait_and_front(T& result);
    bool empty() const;
    int size() const;
    int frames{0};
    int packets{0};
    int frame_lost{0};
    int packet_lost{0};
private:
    mutable std::mutex mtx_;
    std::queue<T> queue_;
    std::condition_variable cond_;
    int max_queue_size{10}; //设置队列最大缓冲值，目前设置最大为10
};

template<typename T>
void safe_queue<T>::push(const T& value) {
    std::lock_guard<std::mutex> lock(mtx_);
    queue_.push(value);
    if constexpr (std::is_same<T, T_Packet>::value) {
        packets ++;
    }
    if constexpr (std::is_same<T, T_Frame>::value) {
        frames ++;
    }
    if(queue_.size() >= max_queue_size) { 
        if constexpr (std::is_same<T, T_Packet>::value) {
            packet_lost ++;
            T_Packet pkt = queue_.front();
            av_packet_unref(pkt.first);
        }
        if constexpr (std::is_same<T, T_Frame>::value) {
            frame_lost ++;
            T_Frame frame = queue_.front();
            av_frame_unref(frame.first);
        }
        queue_.pop();
    }
    cond_.notify_one();
}

template<typename T>
bool safe_queue<T>::try_pop(T& result) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (queue_.empty())
        return false;
    result = std::move(queue_.front());
    queue_.pop();
    return true;
}

template<typename T>
void safe_queue<T>::wait_and_pop(T& result) {
    std::unique_lock<std::mutex> lock(mtx_);
    cond_.wait(lock, [this] { return !queue_.empty(); });
    result = std::move(queue_.front());
    queue_.pop();
}

template<typename T>
void safe_queue<T>::wait_and_front(T& result) {
    std::unique_lock<std::mutex> lock(mtx_);
    cond_.wait(lock, [this] { return !queue_.empty(); });
    result = std::move(queue_.front());
}

template<typename T>
bool safe_queue<T>::empty() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return queue_.empty();
}

template<typename T>
int safe_queue<T>::size() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return queue_.size();
}