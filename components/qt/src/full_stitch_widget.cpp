// full_stitch_widget.cpp
#include "full_stitch_widget.h"

#include <memory>

#include <QDebug>
#include <QLoggingCategory>
#include <QMetaObject>

extern "C" {
    #include <libavutil/hwcontext.h>
    #include <libavutil/pixdesc.h>
}

#include "log.hpp"
#include "tools.hpp"

void* full_stitch_widget::aligned_alloc(size_t size, size_t alignment) {
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

void full_stitch_widget::aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

full_stitch_widget::full_stitch_widget(QWidget *parent) : 
    QOpenGLWidget(parent),
    m_render(nullptr),
    cam(nullptr),
    m_width(0),
    m_height(0),
    m_y_stride(0),
    m_uv_stride(0)
{
    setFixedSize(20803,2160);
    QLoggingCategory::setFilterRules("*.debug=false\n*.warning=false");
    m_render = new Nv12Render();
    cpu_frame = av_frame_alloc();
    cam = camera_manager::GetInstance();
    auto callback_handle = std::bind(&full_stitch_widget::consumerThread, this, std::placeholders::_1);
    cam->setStitchStreamCallback(0, callback_handle);
}

full_stitch_widget::~full_stitch_widget() {
    cleanup();
}

void full_stitch_widget::cleanup() {
    av_frame_free(&cpu_frame);
    if (m_render) {
        delete m_render;
        m_render = nullptr;
    }
}

void full_stitch_widget::initializeGL() {
    if (m_render) {
        m_render->initialize();
    }
}

void full_stitch_widget::paintGL() {
    if (!m_buffer.empty() && m_width > 0 && m_height > 0) {
        m_render->render(m_buffer.data(), m_width, m_height, m_y_stride, m_uv_stride);
    }
}

void full_stitch_widget::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
}

void full_stitch_widget::consumerThread(Frame frame) {   
    AVFrame* src_frame = frame.m_data;
    AVFrame* process_frame = nullptr;
    
    // std::unique_lock<std::mutex> lock(m_mutex, std::try_to_lock);
        // if (!lock.owns_lock()) {
        //     av_frame_free(&process_frame);
        //     continue;
        // }

    // 硬件帧转换到CPU
    if (src_frame->format == AV_PIX_FMT_CUDA) {
        if (av_hwframe_transfer_data(cpu_frame, src_frame, 0) < 0) {
            qWarning() << "Failed to transfer frame to CPU";
            av_frame_free(&frame.m_data);
            return;
        }
        process_frame = cpu_frame;
    }

    m_width = process_frame->width;
    m_height = process_frame->height;
    m_y_stride = process_frame->linesize[0];
    m_uv_stride = process_frame->linesize[1];
    
    draw_vertical_line_nv12(process_frame, 200, "-120°", 150, 0);
    draw_vertical_line_nv12(process_frame, 5350, "-60°", 150, 0);
    draw_vertical_line_nv12(process_frame, 10500, "0°", 150, 0);
    draw_vertical_line_nv12(process_frame, 15550, "60°", 150, 0);
    draw_vertical_line_nv12(process_frame, 20600, "120°", 360, 0);
    
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
    

    // std::unique_lock<std::mutex> lock(m_mutex, std::try_to_lock);
    // if (!lock.owns_lock()) {
    //     av_frame_free(&process_frame);
    //     continue;
    // }
    if (m_buffer.size() < total_size) {
        m_buffer.resize(total_size);
    }
    memcpy(m_buffer.data(), process_frame->data[0], process_frame->linesize[0] * m_height);
    memcpy(m_buffer.data() + y_size, process_frame->data[1], process_frame->linesize[1] * (m_height / 2));
    av_frame_free(&frame.m_data);
    QMetaObject::invokeMethod(this, "update", Qt::QueuedConnection);
}
