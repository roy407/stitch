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
    cam(nullptr),
    con(nullptr),
    running(true),
    m_width(0),
    m_height(0),
    m_y_stride(0),
    m_uv_stride(0),
    last_title_update(std::chrono::steady_clock::now()),
    update_interval(std::chrono::seconds(1)) 
{
    // 设置最小尺寸，允许窗口缩放（移除固定尺寸限制）
    setFixedSize(640, 360);  // 最小尺寸：360p
    
    QLoggingCategory::setFilterRules("*.debug=false\n*.warning=false");
    m_render = new Nv12Render();
    // 只获取实例，不启动（由主窗口统一管理）
    cam = camera_manager::GetInstance();
    
    // 获取红外拼接流
    q = &(cam->get_stitch_IR_camera_stream());
    
    con = QThread::create([this](){consumerThread();});
    con->start();
}

InfraredWidget::~InfraredWidget() {
    cleanup();
}

void InfraredWidget::cleanup() {
    running.store(false);
    if (con) {
        q->stop();
        con->wait();
        delete con;
        con = nullptr;
        LOG_DEBUG("infrared widget consumer thread destroyed!");
    }
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
    QString title = QString("红外拼接-耗时 %1ms").arg(cost_time, 0, 'f', 2);
    emit IRTitle(title);
}
//红外拼接帧的处理
void InfraredWidget::consumerThread() {
    AVFrame* cpu_frame = av_frame_alloc();    
    
    while (running.load()) {
        Frame frame;
        if(!q->wait_and_pop(frame)) break;

        std::unique_lock<std::mutex> lock(m_mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            av_frame_free(&frame.m_data);
            continue;
        }

        AVFrame* src_frame = frame.m_data;
        AVFrame* process_frame = src_frame;
        
        // 硬件帧转换到CPU
        if (src_frame->format == AV_PIX_FMT_CUDA) {
            if (av_hwframe_transfer_data(cpu_frame, src_frame, 0) < 0) {
                qWarning() << "Failed to transfer frame to CPU";
                av_frame_free(&frame.m_data);
                continue;
            }
            process_frame = cpu_frame;
        }

        m_width = process_frame->width;
        m_height = process_frame->height;
        m_y_stride = process_frame->linesize[0];
        m_uv_stride = process_frame->linesize[1];

        frame.m_costTimes.when_show_on_the_screen = get_now_time();
        double dec_to_stitch = 0.0;
        if (frame.m_costTimes.when_get_packet[8] != 0) {
            dec_to_stitch = (frame.m_costTimes.when_show_on_the_screen - 
                                        frame.m_costTimes.when_get_packet[8]) * 1e-6;
        }
        
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
        
        // 复制Y和UV数据（红外相机也使用NV12/YUV格式）
        memcpy(m_buffer.data(), process_frame->data[0], process_frame->linesize[0] * m_height);
        memcpy(m_buffer.data() + y_size, process_frame->data[1], process_frame->linesize[1] * (m_height / 2));
       
        av_frame_free(&frame.m_data);

        auto now = std::chrono::steady_clock::now();
        if (now - last_title_update >= update_interval) {
        QMetaObject::invokeMethod(this, "IRTitleTime", Qt::QueuedConnection, 
                                Q_ARG(double, dec_to_stitch));
        last_title_update = now;  
        }
        
        QMetaObject::invokeMethod(this, "update", Qt::QueuedConnection);
    }
    q->clear();
 
    av_frame_free(&cpu_frame);
}

