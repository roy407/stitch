// infrared_camera_widget.cpp
// 红外拼接显示组件
// 功能：从camera_manager获取红外拼接流并显示
// 配置：使用config::GetIRCameraConfig()和config::GetIRStitchConfig()
#include "infrared_camera_widget.h"
#include <QDebug>
#include <QMetaObject>
#include <memory>
#include <QLoggingCategory>
extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
}
#include "log.hpp"
#include "config.h"
#include "tools.hpp"
//红外拼接初始化
InfraredWidget::InfraredWidget(QWidget *parent) : 
    QOpenGLWidget(parent),
    m_render(nullptr),
    m_width(0),
    m_height(0),
    m_y_stride(0),
    m_uv_stride(0),
    last_title_update(std::chrono::steady_clock::now()),
    update_interval(std::chrono::seconds(1)) 
{
    // 设置最小尺寸，允许窗口缩放（移除固定尺寸限制）
    setMinimumSize(640, 360);  // 最小尺寸：360p
    
    QLoggingCategory::setFilterRules("*.debug=false\n*.warning=false");
    m_render = new Nv12Render();
    cpu_frame = av_frame_alloc();  
    
    // [MODIFIED BEGIN] - 初始化SHM接收器
    m_shm_receiver = new ShmReceiver("/stitch_view_shm_ir");
    if (!m_shm_receiver->init()) {
        qWarning() << "InfraredWidget: Failed to init ShmReceiver!";
    } else {
        qDebug() << "InfraredWidget: ShmReceiver validated.";
    }
    
    m_running = true;
    m_recv_thread = std::thread(&InfraredWidget::shmLoop, this);
    // [MODIFIED END]
}

InfraredWidget::~InfraredWidget() {
    // [MODIFIED BEGIN]
    m_running = false;
    if (m_recv_thread.joinable()) {
        m_recv_thread.join();
    }
    if (m_shm_receiver) {
        delete m_shm_receiver; 
        m_shm_receiver = nullptr;
    }
    // [MODIFIED END]
    cleanup();
}

void InfraredWidget::cleanup() {
    av_frame_free(&cpu_frame);
    if (m_render) {
        delete m_render;
        m_render = nullptr;
    }
}

void InfraredWidget::initializeGL() {
    if (m_render) {
        m_render->initialize();
    }
}

void InfraredWidget::paintGL() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_buffer.empty() && m_width > 0 && m_height > 0) {
        m_render->render(m_buffer.data(), m_width, m_height, m_y_stride, m_uv_stride);
    }
}

void InfraredWidget::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
}
void InfraredWidget::IRTitleTime(double cost_time){
    QString title = QString("红外拼接");
    emit IRTitle(title);
}

// [MODIFIED BEGIN] - SHM Loop
void InfraredWidget::shmLoop() {
    AVFrame* frame = av_frame_alloc();
    while (m_running) {
        if (m_shm_receiver && m_shm_receiver->recvFrame(frame, 100)) {
            processFrame(frame);
            av_frame_unref(frame);
        }
    }
    av_frame_free(&frame);
}

void InfraredWidget::processFrame(AVFrame* process_frame) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    m_width = process_frame->width;
    m_height = process_frame->height;
    m_y_stride = process_frame->linesize[0];
    m_uv_stride = process_frame->linesize[1];
    
    // 确保行对齐是32字节的倍数
    if (m_y_stride % 32 != 0) {
        m_y_stride = ((m_y_stride + 31) / 32) * 32;
    }
    if (m_uv_stride % 32 != 0) {
        m_uv_stride = ((m_uv_stride + 31) / 32) * 32;
    }
    
    size_t y_size = m_y_stride * m_height;
    size_t uv_size = m_uv_stride * (m_height / 2);
    size_t total_size = y_size + uv_size;
    
    // 调整缓冲区大小
    if (m_buffer.size() < total_size) {
        m_buffer.resize(total_size);
    }
    
    // 复制Y和UV数据
    memcpy(m_buffer.data(), process_frame->data[0], process_frame->linesize[0] * m_height);
    memcpy(m_buffer.data() + y_size, process_frame->data[1], process_frame->linesize[1] * (m_height / 2));
    
    // 更新标题 (这里暂时无法获取cost_time，传入0或移除相关逻辑)
    // 如果需要保留心跳更新标题的功能:
    auto now = std::chrono::steady_clock::now();
    if (now - last_title_update >= update_interval) {
        QMetaObject::invokeMethod(this, "IRTitleTime", Qt::QueuedConnection, 
                                Q_ARG(double, 0.0));
        last_title_update = now;  
    }
    
    QMetaObject::invokeMethod(this, "update", Qt::QueuedConnection);
}
// [MODIFIED END]

//红外拼接帧的处理 (Old Callback - Disable)
void InfraredWidget::consumerThread(Frame frame) {  
    // 释放可能传入的AVFrame，防止内存泄漏，但不再进行渲染
    if (frame.m_data) {
       av_frame_free(&frame.m_data);
    }
}

