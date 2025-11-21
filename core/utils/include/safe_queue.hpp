#pragma once

/*---- C++模版必须全部写在头文件中！！！ -----*/
/*---- 如果要写在cpp文件中，那么只有预设实例可以使用 -----*/

#include <queue>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include "tools.hpp"
template<typename T>
class safe_queue {
public:
    safe_queue() = default;
    safe_queue(const safe_queue&) = delete;
    safe_queue& operator=(const safe_queue&) = delete;

    void push(const T& value);
    bool try_pop(T& result);
    bool wait_and_pop(T& result);
    bool wait_and_front(T& result);
    bool empty() const;
    int size() const;
    void pop_and_free();
    void clear();
    void stop();
    int frames{0};
    int packets{0};
    int frame_lost{0};
    int packet_lost{0};
private:
    mutable std::mutex mtx_;
    std::queue<T> queue_;
    std::condition_variable cv;
    std::atomic_bool isStop{false};
    int max_queue_size{10}; //设置队列最大缓冲值，目前设置最大为10
};

template<typename T>
void safe_queue<T>::push(const T& value) {
    std::lock_guard<std::mutex> lock(mtx_);
    queue_.push(value);
    if constexpr (std::is_same<T, Packet>::value) {
        packets ++;
    }
    if constexpr (std::is_same<T, Frame>::value) {
        frames ++;
    }
    if(queue_.size() >= max_queue_size) { 
        frame_lost ++;
        pop_and_free();
    }
    cv.notify_one();
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
bool safe_queue<T>::wait_and_pop(T& result) {
    std::unique_lock<std::mutex> lock(mtx_);
    cv.wait(lock, [this] { return isStop || !queue_.empty(); });
    if(isStop && queue_.empty()) return false;
    result = std::move(queue_.front());
    queue_.pop();
    return true;
}

template<typename T>
bool safe_queue<T>::wait_and_front(T& result) {
    std::unique_lock<std::mutex> lock(mtx_);
    cv.wait(lock, [this] { return isStop || !queue_.empty(); });
    if(isStop && queue_.empty()) return false;
    result = std::move(queue_.front());
    return true;
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

template <typename T>
inline void safe_queue<T>::pop_and_free() {
    if constexpr (std::is_same<T, Packet>::value) {
        Packet pkt = queue_.front();
        av_packet_free(&pkt.m_data);
    }
    if constexpr (std::is_same<T, Frame>::value) {
        Frame frame = queue_.front();
        av_frame_free(&frame.m_data);
    }
    queue_.pop();
}

template <typename T>
inline void safe_queue<T>::clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    while(!queue_.empty()) pop_and_free();
}

template <typename T>
inline void safe_queue<T>::stop() {
    isStop.store(true);
    cv.notify_all();
}
