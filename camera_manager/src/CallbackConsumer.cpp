#include "CallbackConsumer.h"

#include <atomic>
#include <fstream>
#include <mutex>

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
}

#include "log.hpp"
#include "tools.hpp"
static std::atomic<bool> s_isFirstWrite{true};
static std::mutex s_watcher_mutex;;
std::ofstream CallbackConsumer::createFile() {
    std::string filename = pipelineName + std::string("_") + get_current_time_filename(".csv");
    {
        std::lock_guard<std::mutex> lock(s_watcher_mutex);
        std::ios_base::openmode mode = std::ios::app;
        if (s_isFirstWrite) {
            mode = std::ios::trunc;
            s_isFirstWrite = false;
        }
        std::ofstream watcher("timingwatcher.txt",mode);
        if (watcher.is_open()) {
        watcher << filename << std::endl;
        }   
    }
    std::ofstream ofs(filename, std::ios::app);  // 追加写入
    if (!ofs.is_open()) {
        LOG_ERROR("Failed to open file: {}", filename);
    }
    return ofs;
}

void CallbackConsumer::setChannel(FrameChannel *channel)
{
    m_channel = channel;
}

void CallbackConsumer::setCallback(Callback_Handle callback) {
    m_callback = callback;
}

CallbackConsumer::CallbackConsumer() {
    m_name += "callback";
    m_callback = [](Frame frame) -> void {
        av_frame_free(&frame.m_data);
    };
}

void CallbackConsumer::setPipelineName(std::string name) {
    pipelineName = name;
}

void CallbackConsumer::setTimingWatcher(bool enable) {
    openTimingWatcher = enable;
}

CallbackConsumer::~CallbackConsumer() {

}

void CallbackConsumer::start() {
    TaskManager::start();
}

void CallbackConsumer::stop() {
    m_channel->stop();
    TaskManager::stop();
}

void CallbackConsumer::run() {
    if (openTimingWatcher) {
        std::ofstream ofs = createFile();
        while(running) {
            Frame frame;
            if(!m_channel->recv(frame)) break;
            m_callback(frame);
            frame.m_costTimes.when_show_on_the_screen = get_now_time();
            save_cost_table_csv(frame.m_costTimes, ofs);
        }
    } else {
        while(running) {
            Frame frame;
            if(!m_channel->recv(frame)) break;
            m_callback(frame);
        }
    }
    m_channel->clear();
}
