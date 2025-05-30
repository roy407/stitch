#include <vector>
#include <mutex>
#include <atomic>
#include <iostream>
#include <queue>
#include <chrono>
#include <cuda_runtime.h>

extern "C" {
    #include "libavformat/avformat.h"
    #include "libavcodec/avcodec.h"
    #include "libavutil/pixfmt.h" 
    #include "libavutil/pixdesc.h" 
    #include "libavutil/opt.h"
    #include "libavutil/log.h"
    #include "libavcodec/bsf.h"
}
#include "safe_queue.hpp"
#include "rtsp.h"
#include "Stitch.h"
#include "image_encoder.h"
#include "tools.hpp"

#define MYIP "127.0.0.1"

bool is_log_print = true;

#define SIZE (5)

void AVFrame_log(const char* cam_name, const AVFrame* frame);
std::string push_stream_stitch_url = "rtsp://" MYIP ":8554/stitch";

safe_queue<AVPacket*> packet_out;

std::atomic<bool> running{true}; // 全局运行标志

void test_stitch_images(const std::string& url) {

    const int width = 640;
    const int height = 360;

    const int cam = 1;

    std::string input_url = "../1.yuv";
    AVFormatContext* out_ctx = nullptr;
    avformat_alloc_output_context2(&out_ctx, nullptr, "rtsp", url.c_str());

    AVStream* out_stream = avformat_new_stream(out_ctx, nullptr);
    out_stream->id = out_ctx->nb_streams - 1; // 设置流ID

    // 3. 配置视频流参数
    AVCodecParameters* codecpar = out_stream->codecpar;
    codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    codecpar->codec_id = AV_CODEC_ID_H264;   
    codecpar->width = width * cam;                  
    codecpar->height = width;                 
    codecpar->format = AV_PIX_FMT_CUDA;   

    out_stream->time_base = (AVRational){1, 20}; 
    safe_queue<AVFrame*> stitched_frames;
    Stitch stitch(cam,width,height);
    image_encoder img_enc(cam*width,height,stitched_frames,packet_out);
    rtsp_server rtsp(packet_out);
    rtsp.start_rtsp_server(&codecpar,&out_stream->time_base,url.c_str());
    img_enc.start_image_encoder();
    int cnt = 0;
    AVFrame* out_image = nullptr;
    AVFrame* inputs[SIZE] = {};
    {   //在此处设置输入
        const AVCodec* codec = avcodec_find_encoder_by_name("h264_nvenc");
        AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
        codec_ctx->width = width;
        codec_ctx->height = height;
        codec_ctx->pix_fmt = AV_PIX_FMT_CUDA;

        AVBufferRef* hw_frames_ctx = create_cuda_hwframe_ctx(codec_ctx, width, height);

        for(int i = 0;i < cam;i ++) {
            inputs[i] =  load_nv12_to_gpu_frame("/home/eric/文档/save/1.yuv",hw_frames_ctx,width,height);
        }
    }
    while(running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        out_image = stitch.do_stitch(inputs);
        if(out_image) stitched_frames.push(out_image);
    }
    std::cout<<__func__<<" exit!"<<std::endl;
}

void cout_message() {
    int device_id = 0;
    size_t free_mem = 0, total_mem = 0;
    while (running) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        cudaSetDevice(device_id);  // 选择 GPU
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cout << "GPU " << device_id << " memory: "
                << (total_mem - free_mem) / (1024.0 * 1024.0) << " MB used / "
                << total_mem / (1024.0 * 1024.0) << " MB total" << std::endl;
    }
    std::cout<<__func__<<" exit!"<<std::endl;
}

void AVFrame_log(const char* cam_name, const AVFrame* frame) {
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

int main() {
    avformat_network_init(); // 初始化网络模块
    
    std::vector<std::thread> workers;

    workers.emplace_back(test_stitch_images, push_stream_stitch_url);

    if(is_log_print)
        workers.emplace_back(cout_message);
    
    for(auto& t : workers) {
        if(t.joinable()) t.join();
    }
    avformat_network_deinit();
    std::cout<<__func__<<" exit!"<<std::endl;
    return 0;
}