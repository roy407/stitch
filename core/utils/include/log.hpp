#pragma once
#include <iostream>
#include <fstream>
extern "C" {
    #include <libavutil/frame.h>
    #include <libavutil/hwcontext.h>
    #include <libswscale/swscale.h>
}
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h> 
#include "config.h"

#define LOG_DEBUG(...) \
    __LOGGER__::GetInstance()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, \
                spdlog::level::debug, __VA_ARGS__)

#define LOG_INFO(...) \
    __LOGGER__::GetInstance()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, \
                spdlog::level::info, __VA_ARGS__)

#define LOG_WARN(...) \
    __LOGGER__::GetInstance()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, \
                spdlog::level::warn, __VA_ARGS__)

#define LOG_ERROR(...) \
    __LOGGER__::GetInstance()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, \
                spdlog::level::err, __VA_ARGS__)

class __LOGGER__ {
private:
    __LOGGER__() {
        console = spdlog::stdout_color_mt("console");
        std::string str = "debug";
        std::ifstream infile(config::GetConfigFileName());
        if (infile.is_open()) {
            json j;
            infile >> j;
            if(j.contains("global") && j["global"].contains("loglevel")) {
                str = j["global"]["loglevel"];
                std::cout<<"Log Level : "<<str<<std::endl;
            } else {
                std::cout<<"Log Level not set , use default : debug"<<std::endl;
            }
        }
        spdlog::level::level_enum level_num = spdlog::level::info;
        if (str == "debug") {
            level_num = spdlog::level::debug;
        } else if (str == "info") {
            level_num = spdlog::level::info;
        } else if (str == "warn") {
            level_num = spdlog::level::warn;
        } else if (str == "error") {
            level_num = spdlog::level::err;
        } else if (str == "critical") {
            level_num = spdlog::level::critical;
        } else {
            level_num = spdlog::level::info;
        }
        console->set_pattern("[%Y-%m-%d %H:%M:%S.%e][%n][%^%l%$][%s:%#][pid:%t] %v");
        console->set_level(level_num);
        spdlog::set_default_logger(console);
    };
    ~__LOGGER__() {};
    std::shared_ptr<spdlog::logger> console;
public:
    static auto GetInstance() {
        static __LOGGER__ __logger__;
        return __logger__.console;
    }
};

inline void AVFrame_log(const char* cam_name, const AVFrame* frame) {
    if (frame) {
        std::cout << "========= AVFrame Info (" << cam_name << ") =========" << std::endl;
        std::cout << "Format:         " << frame->format << std::endl;
        std::cout << "Width x Height: " << frame->width << " x " << frame->height << std::endl;
        std::cout << "PTS:            " << frame->pts << std::endl;
        std::cout << "DTS:            " << frame->pkt_dts << std::endl;
        std::cout << "HW FramesCtx:   " << (frame->hw_frames_ctx ? "Yes (GPU frame)" : "No (CPU frame)") << std::endl;
        
        // 输出前 3 个 data/buf 指针状态
        for (int i = 0; i < 3; ++i) {
            std::cout << "data[" << i << "]:      " << static_cast<const void*>(frame->data[i]) << std::endl;
            std::cout << "linesize[" << i << "]:  " << frame->linesize[i] << std::endl;
        }

        std::cout << "==========================================================" << std::endl;
    }
}

#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        LOG_ERROR("CUDA Error : {}", cudaGetErrorString(e)); \
        return false; \
    } \
} while(0)
