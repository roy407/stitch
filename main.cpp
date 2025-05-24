#include <vector>
#include <mutex>
#include <atomic>
#include <iostream>
#include <queue>
#include <chrono>

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/pixfmt.h> 
    #include <libavutil/pixdesc.h> 
    #include <libavutil/opt.h>
    #include <libavutil/log.h>
    #include <libavcodec/bsf.h>
}
#include "safe_queue.hpp"
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
#define IS_PUSH_STREAM
//将拉流得到的rtsp数据保存在save_rtsp_data_path中 
// #define SAVE_RTSP_DATA
const int save_rtsp_data_time = 10;
const std::string save_rtsp_data_path = "/home/eric/文档/mp4/";
// 数据回灌 打开此项后，可以不从RTSP中读流，转而从文件中读取。
// #define DATA_REFEED

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

safe_queue<AVPacket*> packet_queues[SIZE];
double camera_timestamp[SIZE] = {0.0};
int camera_fps[SIZE] = {0};
std::pair<int,int> camera_res[SIZE];
struct Camera_param camera_para[SIZE];

safe_queue<AVFrame*> images[SIZE];
AVFrame* out_image = nullptr;

std::atomic<bool> running{true}; // 全局运行标志

void process_stream(const std::string& url, int cam_id) {
    AVFormatContext* fmt_ctx = avformat_alloc_context();
    #ifndef DATA_REFEED
    AVDictionary* options = nullptr;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "stimeout", "5000000", 0);
    #endif
    int frame_cnt = 0;
    #ifdef IS_PUSH_STREAM
    rtsp_server rtsp(packet_queues[cam_id]);
    #endif
    while(running) {
        #ifndef DATA_REFEED
        int ret = avformat_open_input(&fmt_ctx, url.c_str(), nullptr, &options);
        #else
        std::string cam_path = save_rtsp_data_path + std::to_string(cam_id) + ".mp4";
        int ret = avformat_open_input(&fmt_ctx, cam_path.c_str(), NULL, NULL);
        #endif
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
                #if defined(SAVE_RTSP_DATA) && !defined(DATA_REFEED)
                AVFormatContext *output_ctx = NULL;
                std::string cam_path = save_rtsp_data_path + std::to_string(cam_id) + ".mp4";
                avformat_alloc_output_context2(&output_ctx, NULL, "mp4", cam_path.c_str());
                AVStream *out_stream = avformat_new_stream(output_ctx, NULL);
                avcodec_parameters_copy(out_stream->codecpar, codecpar);
                out_stream->time_base = stream->time_base;
                if (!(output_ctx->oformat->flags & AVFMT_NOFILE)) {
                    if (avio_open(&output_ctx->pb, cam_path.c_str(), AVIO_FLAG_WRITE) < 0) {
                        std::cerr << "Could not open output file\n";
                        return;
                    }
                }
                AVBSFContext* bsf_ctx = nullptr;
                const AVBitStreamFilter* bsf = av_bsf_get_by_name("h264_mp4toannexb");
                av_bsf_alloc(bsf, &bsf_ctx);
                avcodec_parameters_copy(bsf_ctx->par_in, codecpar);
                bsf_ctx->time_base_in = stream->time_base;
                av_bsf_init(bsf_ctx);
                avformat_write_header(output_ctx, NULL);
                auto start_time = std::chrono::steady_clock::now();
                bool first_key_frame_found = false;
                #endif
                AVPacket pkt;
                while(running && av_read_frame(fmt_ctx, &pkt) >= 0) {
                    if(pkt.stream_index == video_stream) {
                        double pts_sec = pkt.pts * av_q2d(stream->time_base);
                        camera_timestamp[cam_id] = pts_sec;
                        frame_cnt ++;
                        camera_fps[cam_id] = frame_cnt / pts_sec;
                        // std::lock_guard<std::mutex> lock(mtx);
                        // img.set_parameter(codecpar);
                        // std::queue<AVFrame*> tmp_q = img.do_decode(&pkt);
                        // while(!tmp_q.empty()) {
                        //     images[cam_id].push(tmp_q.front());
                        //     tmp_q.pop();
                        // }
                        AVPacket* pkt_copy = av_packet_clone(&pkt);
                        pkt_copy->time_base = stream->time_base;
                        std::unique_lock<std::mutex> lock(mtx);
                        packet_queues[cam_id].push(pkt_copy);
                        
                        #ifdef IS_PUSH_STREAM
                        rtsp.start_rtsp_server(&camera_para[cam_id].codecpar,&camera_para[cam_id].time_base, push_stream_urls[cam_id]);
                        #endif
                        #if defined(SAVE_RTSP_DATA) && !defined(DATA_REFEED)
                        if (!first_key_frame_found) {
                            if (pkt.flags & AV_PKT_FLAG_KEY)
                                first_key_frame_found = true;
                            else {
                                av_packet_unref(&pkt);
                                continue;
                            }
                        }

                        pkt.stream_index = out_stream->index;
                        av_bsf_send_packet(bsf_ctx, &pkt);
                        while (av_bsf_receive_packet(bsf_ctx, &pkt) == 0) {
                            av_interleaved_write_frame(output_ctx, &pkt);
                        }
                        auto now = std::chrono::steady_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                        if (elapsed >= save_rtsp_data_time) {
                            #ifdef IS_PUSH_STREAM
                            rtsp.close_rtsp_server();
                            #endif
                            running.store(false);
                        }
                        #endif
                    }
                    av_packet_unref(&pkt);
                }
                #if defined(SAVE_RTSP_DATA) && !defined(DATA_REFEED)
                av_write_trailer(output_ctx);
                if (!(output_ctx->oformat->flags & AVFMT_NOFILE))
                    avio_closep(&output_ctx->pb);
                avformat_free_context(output_ctx);
                #endif
            }
        }
        avformat_close_input(&fmt_ctx);
    }
    std::cout<<__func__<<cam_id<<" exit!"<<std::endl;
}

void process_stitch_images(const std::string& url) {
    Stitch stitch;
    image_encoder img_enc;
    bool is_rtsp_launched = false;
    // std::thread t_rtsp;
    while(running) {
        bool is_vaild = true;
        AVFrame* inputs[SIZE] = {};
        for(int i=0;i<SIZE;i++) {
            if(!images[i].try_pop(inputs[i])) is_vaild = false;
        }
        if(is_vaild == false) continue;
        out_image = stitch.do_stitch(inputs);
        //AVFrame_log(0,out_image);
        // AVPacket* pkt = img_enc.do_encode(out_image);
        for(int i=0;i<SIZE;i++) {
            av_frame_free(&inputs[i]);
        }
    }
    std::cout<<__func__<<" exit!"<<std::endl;
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
    std::cout<<__func__<<" exit!"<<std::endl;
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
    
    std::vector<std::thread> workers;
    rtsp_server::init_mediamtx();

    for(int i=0; i<SIZE; ++i) {
        workers.emplace_back(process_stream, camera_urls[i], i);
    }
    workers.emplace_back(process_stitch_images, push_stream_stitch_url);

    if(is_log_print)
        workers.emplace_back(cout_message);
    
    for(auto& t : workers) {
        if(t.joinable()) t.join();
    }
    avformat_network_deinit();
    std::cout<<__func__<<" exit!"<<std::endl;
    return 0;
}