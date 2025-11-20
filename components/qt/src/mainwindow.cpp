#include "mainwindow.h"
#include <QThread>
#include <QDebug>
#include <QtConcurrent/QtConcurrent>
extern "C" {
#include <libavutil/imgutils.h>   // av_image_get_buffer_size 所在
#include <libswscale/swscale.h>   // sws_scale, sws_getContext
#include <libavutil/frame.h>
#include <libavutil/mem.h>        // av_malloc
}
#include "log.hpp"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    cam = camera_manager::GetInstance();
    cam->start();
    setFixedSize(2560, 720);
    videoLabel = new QLabel(this);
    videoLabel->setGeometry(10, 10, 2500, 380);  // 设置位置和大小
    videoLabel->setStyleSheet("background-color: black;");
    
    sf = QThread::create([this]() {
        auto& q = cam->get_stitch_camera_stream();
        static std::string filename = std::string("build/") + get_current_time_filename(".txt");

        std::ofstream ofs(filename, std::ios::app);  // 追加写入
        if (!ofs.is_open()) {
            LOG_ERROR("Failed to open file: {}" ,filename);
            return;
        }
        while (running) {
            Frame frame;
            if(!q.wait_and_pop(frame)) break;
            showFrame(frame.m_data);
            av_frame_free(&(frame.m_data)); 
        }
        q.clear();
        ofs.close();
    });
    sf->start();
}


void MainWindow::showFrame(AVFrame* gpu_frame) {

    // 1. 安全检查
    if (!gpu_frame || !gpu_frame->data[0]) {
        qWarning("Invalid frame received");
        return;
    }

    const int width = gpu_frame->width;
    const int height = gpu_frame->height;
    const AVPixelFormat format = (AVPixelFormat)gpu_frame->format;
    
    // 2. 初始化复用资源
    if (!cpuFrame) {
        cpuFrame = av_frame_alloc();
        rgbFrame = av_frame_alloc();
    }

    // 3. GPU->CPU传输（添加错误处理）
    if (av_hwframe_transfer_data(cpuFrame, gpu_frame, 0) < 0) {
        qWarning("Frame transfer failed");
        return;
    }

    // 4. 检查是否需要重新初始化RGB资源
    if (width != lastWidth || height != lastHeight || format != lastFormat) {
        // 释放旧资源
        av_freep(&rgbBuffer);
        sws_freeContext(swsContext);
        swsContext = nullptr;

        // 创建新资源
        rgbBuffer = (uint8_t*)av_malloc(av_image_get_buffer_size(AV_PIX_FMT_RGB24, width, height, 32));
        av_image_fill_arrays(rgbFrame->data, rgbFrame->linesize, rgbBuffer, 
                            AV_PIX_FMT_RGB24, width, height, 32); // 32字节对齐
        
                            
        // 创建或更新转换上下文
        swsContext = sws_getContext(
                        width, height, (AVPixelFormat)cpuFrame->format,
                        width, height, AV_PIX_FMT_RGB24,
                        SWS_BILINEAR, nullptr, nullptr, nullptr);

        lastWidth = width;
        lastHeight = height;
        lastFormat = format;
    }

    // 5. 色彩空间转换
    if (swsContext) {
        sws_scale(swsContext, cpuFrame->data, cpuFrame->linesize,
                 0, height, rgbFrame->data, rgbFrame->linesize);
    } else {
        qWarning("SwsContext not initialized");
        return;
    }

    // 6. 创建QImage（避免额外拷贝）
    QImage img(rgbFrame->data[0], width, height, rgbFrame->linesize[0], 
              QImage::Format_RGB888, [](void*){}, nullptr);

    // 7. 使用异步方式处理缩放（避免阻塞）
    QSize labelSize = videoLabel->size();
    QtConcurrent::run([this, img, labelSize](){
        // 仅在尺寸不匹配时进行缩放
        QImage scaledImg = (img.size() == labelSize) ? 
                          img.copy() : 
                          img.scaled(labelSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        
        // 使用线程安全方式更新UI
        QMetaObject::invokeMethod(this, [this, scaledImg](){
            if (videoLabel && !scaledImg.isNull()) {
                videoLabel->setPixmap(QPixmap::fromImage(scaledImg));
            }
        }, Qt::QueuedConnection);
    });
}

MainWindow::~MainWindow()
{
    running.store(false);
    cam->stop();
    sws_freeContext(swsContext);
    av_frame_free(&cpuFrame);
    av_frame_free(&rgbFrame);
    av_freep(&rgbBuffer);
}
