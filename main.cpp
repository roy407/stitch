#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <iostream>
#include <queue>

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/pixfmt.h> 
    #include <libavutil/pixdesc.h> 
    #include <libavutil/opt.h>
    #include <libavutil/log.h>
}
#include "rtsp.h"
#include "Stitch.h"
#include "image_decoder.h"
#include "image_encoder.h"

#define MYIP "127.0.0.1"

image_decoder img;
std::mutex mtx;

// 打印log
bool is_log_print = true;
// 推流
bool is_push_stream = true;

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
    "rtsp://" MYIP ":8554/cam0",
    "rtsp://" MYIP ":8554/cam1",
    "rtsp://" MYIP ":8554/cam2",
    "rtsp://" MYIP ":8554/cam3",
    "rtsp://" MYIP ":8554/cam4"
};

void AVFrame_log(int cam_id, const AVFrame* frame);
std::string push_stream_stitch_url = "rtsp://" MYIP ":8554/stitch";

AVPacket* packet_queues[SIZE];
double camera_timestamp[SIZE] = {0.0};
int camera_fps[SIZE] = {0};
std::pair<int,int> camera_res[SIZE];
struct Camera_param camera_para[SIZE];

std::queue<AVFrame*> images[SIZE];
AVFrame* out_image = nullptr;

std::atomic<bool> running{true}; // 全局运行标志

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
                            pkt.dts = pkt.pts;
                        }
                        std::lock_guard<std::mutex> lock(mtx);
                        img.set_parameter(codecpar);
                        std::queue<AVFrame*> tmp_q = img.do_decode(&pkt);
                        while(!tmp_q.empty()) {
                            images[cam_id].push(tmp_q.front());
                            tmp_q.pop();
                        }
                        AVPacket* pkt_copy = av_packet_clone(&pkt);
                        pkt_copy->time_base = stream->time_base;
                        packet_queues[cam_id] = pkt_copy;
                        if(!is_rtsp_launched && is_push_stream) {
                            t_rtsp = rtsp_server(&camera_para[cam_id].codecpar,&camera_para[cam_id].time_base, &packet_queues[cam_id], push_stream_urls[cam_id]);
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

void process_stitch_images(const std::string& url) {
    Stitch stitch;
    //image_encoder img_enc;
    bool is_rtsp_launched = false;
    std::thread t_rtsp;
    while(running) {
        AVFrame* inputs[SIZE] = {};
        for(int i=0;i<SIZE;i++) {
            if(!images[i].empty()) {
                inputs[i] = images[i].front();
            }
        }
        out_image = stitch.do_stitch(inputs);
        // AVPacket* pkt = img_enc.do_encode(out_image);
        for(int i=0;i<SIZE;i++) {
            if(!images[i].empty()) {
                av_frame_free(&images[i].front());
                images[i].pop();
            }
        }
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

void AVFrame_log(int cam_id, const AVFrame* frame) {
if (frame) {
    std::cout << "========= AVFrame Info (cam_id=" << cam_id << ") =========" << std::endl;
    std::cout << "Format:         " << av_get_pix_fmt_name((AVPixelFormat)frame->format) << std::endl;
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

int main() {
    avformat_network_init(); // 初始化网络模块
    
    av_log_set_level(AV_LOG_QUIET);
    std::vector<std::thread> workers;
    //rtsp_server::init_server();

    for(int i=0; i<SIZE; ++i) {
        workers.emplace_back(process_stream, camera_urls[i], i);
    }
    workers.emplace_back(process_stitch_images, push_stream_stitch_url);

    if(is_log_print)
        workers.emplace_back(cout_message);
    
    std::cin.get();
    running = false;
    
    for(auto& t : workers) {
        if(t.joinable()) t.join();
    }
    //rtsp_server::close_server();
    return 0;
}