// widget.cpp
#include "widget.h"
#include <QDebug>
#include <QMetaObject>
#include <memory>
#include <QLoggingCategory>
extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
}
#include "log.hpp"

void* Widget::aligned_alloc(size_t size, size_t alignment) {
    void* ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = nullptr;
    }
#endif
    return ptr;
}

void Widget::aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

Widget::Widget(QWidget *parent) : 
    QOpenGLWidget(parent),
    m_render(nullptr),
    cam(nullptr),
    con(nullptr),
    running(true),
    m_width(0),
    m_height(0),
    m_y_stride(0),
    m_uv_stride(0)
{
    // setFixedSize(2560, 1080);
    QLoggingCategory::setFilterRules("*.debug=false\n*.warning=false");
    m_render = new Nv12Render();
    cam = camera_manager::GetInstance();
    cam->start();
    q = &(cam->get_stitch_stream());
    con = QThread::create([this](){consumerThread();});
    con->start();
}

Widget::~Widget() {
    cleanup();
}

void Widget::cleanup() {
    running.store(false);
    if (con) {
        q->stop();
        con->wait();
        delete con;
        con = nullptr;
        LOG_DEBUG("widget consumer thread destroyed!");
    }
    
    if (cam) {
        cam->stop();
        LOG_DEBUG("camera_service stopped!");
    }
    
    if (m_render) {
        delete m_render;
        m_render = nullptr;
    }
}

void Widget::initializeGL() {
    if (m_render) {
        m_render->initialize();
    }
}

void Widget::paintGL() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_buffer.empty() && m_width > 0 && m_height > 0) {
        m_render->render(m_buffer.data(), m_width, m_height, m_y_stride, m_uv_stride);
    }
}

void Widget::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
}

void Widget::consumerThread() {
    static std::string filename = std::string("build/") + get_current_time_filename(".csv");

    std::ofstream ofs(filename, std::ios::app);  // 追加写入
    if (!ofs.is_open()) {
        LOG_ERROR("Failed to open file: {}", filename);
        return;
    }

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
        
        av_frame_free(&frame.m_data);
        QMetaObject::invokeMethod(this, "update", Qt::QueuedConnection);

        frame.m_costTimes.when_show_on_the_screen = get_now_time();
        save_cost_table_csv(frame.m_costTimes,ofs);
    }
    q->clear();
    av_frame_free(&cpu_frame);
}
