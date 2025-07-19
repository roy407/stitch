#include "camera_manager.h"
#include <vector>
#include <mutex>
#include <atomic>
#include <iostream>
#include <queue>
#include <chrono>

#include <chrono>
#include <fstream>
#include <iomanip>  // for std::put_time
#include <ctime>    // for std::localtime
#include <sstream>

// #define IS_PUSH_STREAM

extern "C" {
    #include "libavformat/avformat.h"
    #include "libavcodec/avcodec.h"
    #include "libavutil/pixfmt.h" 
    #include "libavutil/pixdesc.h" 
    #include "libavutil/opt.h"
    #include "libavutil/log.h"
    #include "libavutil/error.h" 
    #include "libavcodec/bsf.h"
}
#include "safe_queue.hpp"
#include "rtsp.h"
#include "image_decoder.h"
#include "image_encoder.h"
#include "config.h"

camera_manager* camera_manager::GetInstance() {
    static camera_manager cam;
    return &cam;
}

camera_manager::camera_manager() {
    // cam_num = config::GetInstance().GetCameraConfig().cam_num();
}

void camera_manager::get_stream_from_rtsp(int cam_id) {
    AVFormatContext* fmt_ctx = avformat_alloc_context();
    AVDictionary* options = nullptr;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "stimeout", "5000000", 0);
    int frame_cnt = 0;
    #ifdef IS_PUSH_STREAM
    std::string push_stream_url = config::GetInstance().GetCameraConfig()[cam_id].output_url;
    rtsp_server rtsp(packet_input[cam_id]);
    #else
    image_decoder img_decoder(packet_input[cam_id],frame_input[cam_id],cam_id);
    #endif
    std::string url;
    if(config::GetInstance().GetGlobalConfig().use_sub_input) {
        url = config::GetInstance().GetCameraConfig()[cam_id].sub_input_url;
    } else {
        url = config::GetInstance().GetCameraConfig()[cam_id].input_url;
    }
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
                camera_res[cam_id] = {codecpar->width,codecpar->height};
                #ifdef IS_PUSH_STREAM
                rtsp.start_rtsp_server(&camera_para[cam_id].codecpar,&camera_para[cam_id].time_base, push_stream_url);
                #else
                img_decoder.start_image_decoder(codecpar);
                #endif
                AVPacket pkt;
                struct costTimes t;
                while(running && av_read_frame(fmt_ctx, &pkt) >= 0) {
                    if(pkt.stream_index == video_stream) {
                        t.when_get_packet[cam_id] = get_now_time();
                        t.image_idx[cam_id] = frame_cnt;
                        double pts_sec = pkt.pts * av_q2d(stream->time_base);
                        camera_timestamp[cam_id] = pts_sec;
                        frame_cnt ++;
                        camera_fps[cam_id] = frame_cnt / pts_sec;
                        AVPacket* pkt_copy = av_packet_clone(&pkt);
                        packet_input[cam_id].push({pkt_copy,t});
                    }
                    av_packet_unref(&pkt);
                }
            }
        }
        avformat_close_input(&fmt_ctx);
    }
    img_decoder.close_image_decoder();
    std::cout<<__func__<<cam_id<<" exit!"<<std::endl;
}

void camera_manager::get_stream_from_file(int cam_id) {
    AVFormatContext* fmt_ctx = avformat_alloc_context();
    int frame_cnt = 0;
    #ifdef IS_PUSH_STREAM
    std::string push_stream_url = config::GetInstance().GetCameraConfig()[cam_id].output_url;
    rtsp_server rtsp(packet_input[cam_id]);
    #else
    image_decoder img_decoder(packet_input[cam_id],frame_input[cam_id],cam_id);
    #endif
    std::string cam_path = config::GetInstance().GetGlobalConfig().save_rtsp_data_path + std::to_string(cam_id) + ".mp4";
    int ret = avformat_open_input(&fmt_ctx, cam_path.c_str(), NULL, NULL);
    if(ret < 0) {
        std::cerr << "Could not open output file\n";
    }
    if(avformat_find_stream_info(fmt_ctx, nullptr) >= 0) {
        int video_stream = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if(video_stream >= 0) {
            AVStream* stream = fmt_ctx->streams[video_stream];
            AVCodecParameters* codecpar = stream->codecpar;
            camera_para[cam_id].codecpar = codecpar;
            camera_para[cam_id].time_base = stream->time_base;
            camera_res[cam_id] = {codecpar->width,codecpar->height};
            auto start_time = std::chrono::steady_clock::now();
            double start_pts = AV_NOPTS_VALUE;
            AVPacket pkt;
            struct costTimes t;
            
            #ifdef IS_PUSH_STREAM
            rtsp.start_rtsp_server(&camera_para[cam_id].codecpar,&camera_para[cam_id].time_base, push_stream_url);
            #else
            img_decoder.start_image_decoder(codecpar);
            #endif
            while(running && av_read_frame(fmt_ctx, &pkt) >= 0) {
                if(pkt.stream_index == video_stream) {
                    t.when_get_packet[cam_id] = get_now_time();
                    t.image_idx[cam_id] = frame_cnt;
                    double pts_sec = pkt.pts * av_q2d(stream->time_base);
                    if (start_pts == AV_NOPTS_VALUE) {
                        start_pts = pts_sec;
                        start_time = std::chrono::steady_clock::now();
                    }

                    camera_timestamp[cam_id] = pts_sec;
                    frame_cnt ++;
                    camera_fps[cam_id] = frame_cnt / pts_sec;
                    double relative_pts = pts_sec - start_pts;
                    auto target_time = start_time + std::chrono::duration<double>(relative_pts);
                    auto now = std::chrono::steady_clock::now();
                    
                    if (now < target_time) {
                        std::this_thread::sleep_until(target_time);
                    }
                    AVPacket* pkt_copy = av_packet_clone(&pkt);
                    av_packet_rescale_ts(pkt_copy, 
                        stream->time_base,
                        stream->time_base); 
                    packet_input[cam_id].push({pkt_copy,t});
                }
                av_packet_unref(&pkt);
            }
        }
    }
    running.store(false);
    img_decoder.close_image_decoder();
    avformat_close_input(&fmt_ctx);
    std::cout<<__func__<<" "<<cam_id<<" exit!"<<std::endl;
}

void camera_manager::save_stream_to_file(int cam_id) {
    AVFormatContext* fmt_ctx = avformat_alloc_context();
    AVDictionary* options = nullptr;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "stimeout", "5000000", 0);
    int frame_cnt = 0;
    std::string url;
    if(config::GetInstance().GetGlobalConfig().use_sub_input) {
        url = config::GetInstance().GetCameraConfig()[cam_id].sub_input_url;
    } else {
        url = config::GetInstance().GetCameraConfig()[cam_id].input_url;
    }
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
                camera_res[cam_id] = {codecpar->width,codecpar->height};
                AVFormatContext *output_ctx = avformat_alloc_context();
                std::string cam_path = config::GetInstance().GetGlobalConfig().save_rtsp_data_path + std::to_string(cam_id) + ".mp4";
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
                int ret = avformat_write_header(output_ctx, NULL);
                if (ret < 0) {
                    char error_buffer[AV_ERROR_MAX_STRING_SIZE];
                    av_strerror(ret, error_buffer, sizeof(error_buffer));
                    fprintf(stderr, "Could not write header (error '%s')\n", error_buffer);
                    avformat_free_context(output_ctx);
                    return ;
                }
                auto start_time = std::chrono::steady_clock::now();
                bool first_key_frame_found = false;
                AVPacket pkt;
                while(running && av_read_frame(fmt_ctx, &pkt) >= 0) {
                    if(pkt.stream_index == video_stream) {
                        double pts_sec = pkt.pts * av_q2d(stream->time_base);
                        camera_timestamp[cam_id] = pts_sec;
                        frame_cnt ++;
                        camera_fps[cam_id] = frame_cnt / pts_sec;
                        av_interleaved_write_frame(output_ctx, &pkt);
                        auto now = std::chrono::steady_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                        if (elapsed >= config::GetInstance().GetGlobalConfig().save_rtsp_data_time) {
                            running.store(false);
                        }
                    }
                    av_packet_unref(&pkt);
                }
                av_write_trailer(output_ctx);
                if (!(output_ctx->oformat->flags & AVFMT_NOFILE))
                    avio_closep(&output_ctx->pb);
                avformat_free_context(output_ctx);
            }
        }
        avformat_close_input(&fmt_ctx);
    }
    std::cout<<__func__<<cam_id<<" exit!"<<std::endl;
}

// void camera_manager::do_stitch() {
//     int width = 3840;
//     int height = 2160;

//     std::string url = config::GetInstance().GetGlobalStitchConfig().output_url;

//     AVFormatContext* out_ctx = nullptr;
//     avformat_alloc_output_context2(&out_ctx, nullptr, "rtsp", url.c_str());

//     AVStream* out_stream = avformat_new_stream(out_ctx, nullptr);
//     out_stream->id = out_ctx->nb_streams - 1; // 设置流ID

//     // 3. 配置视频流参数
//     AVCodecParameters* codecpar = out_stream->codecpar;
//     codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
//     codecpar->codec_id = AV_CODEC_ID_H264;   
//     codecpar->width = width * cam_num;                  
//     codecpar->height = height;                 
//     codecpar->format = AV_PIX_FMT_CUDA;   

//     out_stream->time_base = (AVRational){1, 20}; 
//     Stitch stitch(width,height,cam_num);
//     // image_encoder img_enc(width * cam_num,height,frame_output,packet_output);
//     // rtsp_server rtsp(packet_output);
//     // rtsp.start_rtsp_server(&codecpar,&out_stream->time_base,url.c_str());
//     // img_enc.start_image_encoder();
//     int cnt = 0;
//     AVFrame* out_image = nullptr;

//     while (running) {
//         AVFrame* inputs[cam_num] = {};
//         std::pair<AVFrame*,costTimes> inputs_[cam_num];
//         for (int i = 0; i < cam_num; i++) {
//             frame_input[i].wait_and_pop(inputs_[i]);
//             inputs[i] = inputs_[i].first;
//         }
//         out_image = stitch.do_stitch(inputs);
//         costTimes t;
//         for (int i=0;i < cam_num; i++) {
//             t.image_idx[i] = inputs_[i].second.image_idx[i];
//             t.when_get_packet[i] = inputs_[i].second.when_get_packet[i];
//             t.when_get_decoded_frame[i] = inputs_[i].second.when_get_decoded_frame[i];
//         }
//         t.when_get_stitched_frame = get_now_time();
//         out_image->pts = inputs[0]->pts;
//         frame_output.push({out_image,t});
//         for (int i = 0; i < cam_num; ++i) {
//             if (inputs[i]) {
//                 av_frame_free(&inputs[i]);
//             }
//         }
//     }
//     std::cout<<__func__<<" exit!"<<std::endl;
// }

void camera_manager::start() {
    avformat_network_init(); // 初始化网络模块

    const std::string software_status = config::GetInstance().GetGlobalConfig().software_status;

    if(software_status == "release") {
        rtsp_server::init_mediamtx();
    }
    const std::string status = config::GetInstance().GetGlobalConfig().status;
    for(int i=0; i<cam_num; ++i) {
        if(status == "file")
                workers.emplace_back(&camera_manager::get_stream_from_file, this, i);
        else if(status == "save")
                workers.emplace_back(&camera_manager::save_stream_to_file, this, i);
        else if(status == "rtsp")
                workers.emplace_back(&camera_manager::get_stream_from_rtsp, this, i);
    }
    if(status != "save") {
        workers.emplace_back(&camera_manager::do_stitch,this);
    }

    workers.emplace_back(&camera_manager::cout_message,this);

    if(software_status == "release") {
        rtsp_server::destory_mediamtx();
    }
}

void camera_manager::stop() {
    running = false;
    for(auto& w: workers) {
        if(w.joinable()) {
            w.join();
        }
    }
    avformat_network_deinit();
}

safe_queue<std::pair<AVFrame*,costTimes>>& camera_manager::get_stitch_stream() {
    return frame_output;
}

void camera_manager::cout_message() {
    int device_id = 0;
    size_t free_mem = 0, total_mem = 0;
    #if 1
    std::vector<int> last_frame_counts(cam_num, 0);
    std::vector<int> last_packet_counts(cam_num, 0);
    int last_global_frame = 0;
    int last_global_packet = 0;
    #endif
    while (running) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::cout << std::endl;
        std::cout << std::endl;
        for(int cam_id = 0;cam_id < cam_num;cam_id ++) {
            std::cout <<" cam_id "<< '[' << cam_id << ']' 
            << "  res:" << '[' << camera_res[cam_id].first << ',' << camera_res[cam_id].second << ']' 
            << "  timestamp:" << camera_timestamp[cam_id] 
            << "  FPS:" << camera_fps[cam_id] << std::endl;

        }
        
        aclError set_device_ret = aclrtSetDevice(device_id);
        if (set_device_ret != ACL_SUCCESS) {
            const char* err_msg = aclGetRecentErrMsg();
            std::cerr << "Set device failed: " << err_msg << std::endl;
            continue;
        }
        
        #if 1 
        // Ascend 310芯片
        aclError mem_ret = aclrtGetMemInfo(ACL_DDR_MEM, &free_mem, &total_mem);
        #else
        // Ascend 910芯片
        aclError mem_ret = aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem);
        #endif
        
        if (mem_ret != ACL_SUCCESS) {
            const char* err_msg = aclGetRecentErrMsg();
            std::cerr << "Get memory info failed: " << err_msg << std::endl;
        } else {
            // 打印内存信息
            std::cout << "NPU " << device_id << " memory: "
                      << "Free: " << free_mem / (1024 * 1024) << "MB, "
                      << "Total: " << total_mem / (1024 * 1024) << "MB" << std::endl;
        }

        #if 1
        std::cout << "=== Per-Camera Input Stats ===\n";
        for (int i = 0; i < cam_num; ++i) {
            int current_packets = packet_input[i].packets;
            int current_frames  = frame_input[i].frames;

            std::cout << "[Cam " << i << "] "
                    << "packet_in lost: " << packet_input[i].packet_lost << " / "
                    << "total: " << current_packets
                    << " --- speed: " << (current_packets - last_packet_counts[i]) / 2 << "\n";

            std::cout << "[Cam " << i << "] "
                    << "frame_in  lost: " << frame_input[i].frame_lost << " / "
                    << "total: " << current_frames
                    << " --- speed: " << (current_frames - last_frame_counts[i]) / 2 << "\n";

            last_packet_counts[i] = current_packets;
            last_frame_counts[i]  = current_frames;
        }

        std::cout << "\n=== Global Output Stats ===\n";
        int curr_out_frame = frame_output.frames;
        int curr_out_packet = packet_output.packets;

        std::cout << "frame_out lost: " << frame_output.frame_lost << " / "
                << "total: " << curr_out_frame
                << " --- speed: " << (curr_out_frame - last_global_frame) / 2 << "\n";

        std::cout << "packet_out lost: " << packet_output.packet_lost << " / "
                << "total: " << curr_out_packet
                << " --- speed: " << (curr_out_packet - last_global_packet) / 2 << "\n";

        last_global_frame = curr_out_frame;
        last_global_packet = curr_out_packet;
        #endif
        }
    std::cout<<__func__<<" exit!"<<std::endl;
}