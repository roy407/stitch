#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <iostream>
#include <queue>

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
}

// 打印log
bool is_log_print = true;
// 推流
bool is_push_stream = false;

#define SIZE (5)

struct Camera_param {
    AVCodecParameters* codecpar; 
    AVRational time_base;
};

std::vector<std::string> camera_urls = {
    "rtsp://admin:ky406sys@192.168.1.50:554/h265/ch1/main/av_stream",
    "rtsp://admin:ky406sys@192.168.1.51:554/h265/ch1/main/av_stream",
    "rtsp://admin:ky406sys@192.168.1.52:554/h265/ch1/main/av_stream",
    "rtsp://admin:ky406sys@192.168.1.53:554/h265/ch1/main/av_stream",
    "rtsp://admin:ky406sys@192.168.1.54:554/h265/ch1/main/av_stream"
};

std::vector<std::string> push_stream_urls = {
    "rtsp://192.168.3.50:8554/cam0",
    "rtsp://192.168.3.50:8554/cam1",
    "rtsp://192.168.3.50:8554/cam2",
    "rtsp://192.168.3.50:8554/cam3",
    "rtsp://192.168.3.50:8554/cam4"
};

AVPacket* packet_queues[SIZE];

double camera_timestamp[SIZE] = {0.0};
int camera_fps[SIZE] = {0};
std::pair<int,int> camera_res[SIZE];
struct Camera_param camera_para[SIZE];


std::atomic<bool> running{true}; // 全局运行标志

void push_stream(int cam_id, const std::string& output_url) {
    if (!camera_para[cam_id].codecpar || camera_para[cam_id].codecpar->codec_type != AVMEDIA_TYPE_VIDEO) {
        std::cout << "Invalid codec parameters" << std::endl;
        return;
    }
    AVFormatContext* out_ctx = nullptr;
    avformat_alloc_output_context2(&out_ctx, nullptr, "rtsp", output_url.c_str());
    AVStream* out_stream = avformat_new_stream(out_ctx, nullptr);
    avcodec_parameters_copy(out_stream->codecpar, camera_para[cam_id].codecpar);
    av_dict_set(&out_ctx->metadata, "rtsp_transport", "tcp", 0); 
    av_dict_set(&out_ctx->metadata, "muxdelay", "0.1", 0);
    av_dict_set(&out_ctx->metadata, "use_wallclock_as_timestamps", "1", 0);
    while(running) {
        AVPacket* pkt = packet_queues[cam_id];
        if(pkt == nullptr) continue;
        pkt->pts = av_rescale_q(pkt->pts, camera_para[cam_id].time_base, out_stream->time_base);
        pkt->dts = av_rescale_q(pkt->dts, camera_para[cam_id].time_base, out_stream->time_base);
        av_interleaved_write_frame(out_ctx, pkt);
        av_packet_unref(pkt);
    }
}

// 线程处理函数
void process_stream(const std::string& url, int cam_id) {
    AVFormatContext* fmt_ctx = avformat_alloc_context();
    AVDictionary* options = nullptr;
    int frame_cnt = 0;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "stimeout", "5000000", 0);
    std::thread t_rtsp;
    bool is_rtsp_launched = false;
    while(running) {
        int ret = avformat_open_input(&fmt_ctx, url.c_str(), nullptr, &options);
        if(ret < 0) {
            std::this_thread::sleep_for(std::chrono::seconds(3));
            continue;
        }
        if(avformat_find_stream_info(fmt_ctx, nullptr) >= 0) {
            int video_stream = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
            if(video_stream >= 0) {
                AVStream* stream = fmt_ctx->streams[video_stream];
                AVCodecParameters* codecpar = stream->codecpar;
                camera_para[cam_id].codecpar = codecpar;
                camera_para[cam_id].time_base = stream->time_base;
                {
                    camera_res[cam_id] = {codecpar->width,codecpar->height};
                }
                AVPacket pkt;
                while(running && av_read_frame(fmt_ctx, &pkt) >= 0) {
                    if(pkt.stream_index == video_stream) {
                        double pts_sec = pkt.pts * av_q2d(stream->time_base);
                        camera_timestamp[cam_id] = pts_sec;
                        frame_cnt ++;
                        camera_fps[cam_id] = frame_cnt / pts_sec;
                        if (pkt.pts == AV_NOPTS_VALUE || pkt.dts == AV_NOPTS_VALUE) {
                            static std::atomic<int64_t> auto_pts{0};
                            const int64_t interval = av_rescale_q(1, 
                                (AVRational){1, 20},
                                stream->time_base);
                            
                            pkt.pts = auto_pts.fetch_add(interval);
                            pkt.dts = pkt.pts; // 保持PTS=DTS
                        }
                        AVPacket* pkt_copy = av_packet_clone(&pkt);
                        pkt_copy->time_base = stream->time_base;
                        packet_queues[cam_id] = pkt_copy;
                        if(!is_rtsp_launched && is_push_stream) {
                            t_rtsp = std::thread(push_stream ,cam_id, std::ref(push_stream_urls[cam_id]));
                            is_rtsp_launched = true;
                        }
                    }
                    av_packet_unref(&pkt);
                }
            }
        }
        avformat_close_input(&fmt_ctx);
    }
}

void cout_message() {
    while (running) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::cout << std::endl;
        std::cout << std::endl;
        for(int cam_id = 0;cam_id < SIZE;cam_id ++) {
            std::cout <<" cam_id "<< '[' << cam_id << ']' 
            << "  res:" << '[' << camera_res[cam_id].first << ',' << camera_res[cam_id].second << ']' 
            << "  timestamp:" << camera_timestamp[cam_id] 
            << "  FPS:" << camera_fps[cam_id] << std::endl;

        }
    }
}

int main() {
    avformat_network_init(); // 初始化网络模块
    
    std::vector<std::thread> workers;
    
    for(int i=0; i<SIZE; ++i) {
        workers.emplace_back(process_stream, camera_urls[i], i);
    }
    if(is_log_print)
        workers.emplace_back(cout_message);
    
    std::cin.get();
    running = false;
    
    for(auto& t : workers) {
        if(t.joinable()) t.join();
    }
    
    return 0;
}