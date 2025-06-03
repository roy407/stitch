#pragma once

/*---- C++模版必须全部写在头文件中！！！ -----*/
/*---- 如果要写在cpp文件中，那么只有预设实例可以使用 -----*/

#include <queue>
#include <mutex>
#include <condition_variable>

template<typename T>
class safe_queue {
public:
    safe_queue() = default;
    safe_queue(const safe_queue&) = delete;
    safe_queue& operator=(const safe_queue&) = delete;

    void push(const T& value);
    bool try_pop(T& result);
    void wait_and_pop(T& result);
    bool empty() const;
    int size() const;
    std::atmoic<int> timestamp;
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
    if constexpr (std::is_same<T, AVPacket*>::value) {
        packets ++;
    }
    if constexpr (std::is_same<T, AVFrame*>::value) {
        frames ++;
    }
    if(queue_.size() >= max_queue_size) { 
        if constexpr (std::is_same<T, AVPacket*>::value) {
            packet_lost ++;
            AVPacket* pkt = queue_.front();
            av_packet_unref(pkt);
        }
        if constexpr (std::is_same<T, AVFrame*>::value) {
            frame_lost ++;
            AVFrame* frame = queue_.front();
            av_frame_unref(frame);
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
bool safe_queue<T>::empty() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return queue_.empty();
}

template<typename T>
int safe_queue<T>::size() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return queue_.size();
}
