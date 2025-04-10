#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <iostream>

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
}

#define SIZE (5)

std::vector<std::string> camera_urls = {
    "rtsp://admin:ky406sys@192.168.1.50:554/h265/ch1/main/av_stream",
    "rtsp://admin:ky406sys@192.168.1.51:554/h265/ch1/main/av_stream",
    "rtsp://admin:ky406sys@192.168.1.52:554/h265/ch1/main/av_stream",
    "rtsp://admin:ky406sys@192.168.1.53:554/h265/ch1/main/av_stream",
    "rtsp://admin:ky406sys@192.168.1.54:554/h265/ch1/main/av_stream"
};

double camera_timestamp[SIZE];
int camera_fps[SIZE];
std::pair<int,int> camera_res[SIZE];

std::mutex cout_mutex; // 控制台输出锁
std::atomic<bool> running{true}; // 全局运行标志

// 线程处理函数
void process_stream(const std::string& url, int cam_id) {
    AVFormatContext* fmt_ctx = avformat_alloc_context();
    AVDictionary* options = nullptr;
    int frame_cnt = 0;
    
    // 设置RTSP参数
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "stimeout", "5000000", 0); // 5秒超时

    // 打开流（带重试机制）
    while(running) {
        int ret = avformat_open_input(&fmt_ctx, url.c_str(), nullptr, &options);
        if(ret < 0) {
            std::this_thread::sleep_for(std::chrono::seconds(3));
            continue;
        }
        
        // 获取流信息
        if(avformat_find_stream_info(fmt_ctx, nullptr) >= 0) {
            // 查找视频流索引
            int video_stream = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
            
            if(video_stream >= 0) {
                AVStream* stream = fmt_ctx->streams[video_stream];
                AVCodecParameters* codecpar = stream->codecpar;
                
                // 输出分辨率
                {
                    camera_res[cam_id] = {codecpar->width,codecpar->height};
                }
                
                // 读取数据包（网页4改进版）
                AVPacket pkt;
                while(running && av_read_frame(fmt_ctx, &pkt) >= 0) {
                    if(pkt.stream_index == video_stream) {
                        // 转换时间戳（网页4）
                        double pts_sec = pkt.pts * av_q2d(stream->time_base);
                        camera_timestamp[cam_id] = pts_sec;
                        frame_cnt ++;
                        camera_fps[cam_id] = frame_cnt / pts_sec;
                    }
                    av_packet_unref(&pkt);
                }
            }
        }
        
        // 清理资源
        avformat_close_input(&fmt_ctx);
    }
}

void cout_message() {
    while (running) {
        for(int cam_id = 0;cam_id < SIZE;cam_id ++) {
            std::cout <<" cam_id "<< '[' << cam_id << ']' 
            << "  res:" << '[' << camera_res[cam_id].first << ',' << camera_res[cam_id].second << ']' 
            << "  timestamp:" << camera_timestamp[cam_id] 
            << "  FPS:" << camera_fps[cam_id] << std::endl;

        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}

int main() {
    avformat_network_init(); // 初始化网络模块（网页4）
    
    std::vector<std::thread> workers;
    
    // 创建线程池
    for(int i=0; i<camera_urls.size(); ++i) {
        workers.emplace_back(process_stream, camera_urls[i], i+1);
    }
    workers.emplace_back(cout_message);
    
    // 等待终止信号
    std::cin.get();
    running = false;
    
    // 清理线程
    for(auto& t : workers) {
        if(t.joinable()) t.join();
    }
    
    return 0;
}