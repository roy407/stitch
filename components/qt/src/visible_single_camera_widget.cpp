#include "visible_single_camera_widget.h"
#include <QVBoxLayout>
#include <QCloseEvent>
#include <QScreen>
#include <QApplication>
#include <QGridLayout>
#include <QLoggingCategory>
#include <QLabel>
#include <QDebug>
#include <QThread>
#include <atomic>
#include <mutex>
#include "config.h"
#include "log.hpp"

extern "C" {
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
}

void* CameraDisplayWidget::aligned_alloc(size_t size, size_t alignment) {
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

void CameraDisplayWidget::aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

CameraDisplayWidget::CameraDisplayWidget(CameraConfig camera_config, QWidget *parent) : 
    QOpenGLWidget(parent),
    m_render(nullptr),
    m_width(0),
    m_height(0),
    m_y_stride(0),
    m_uv_stride(0)
{
    setFixedSize(384,216);
    QLoggingCategory::setFilterRules("*.debug=false\n*.warning=false");
    m_render = new Nv12Render();
    cpu_frame = av_frame_alloc();
    // cam = camera_manager::GetInstance(); // [DECOUPLED]
    // auto handle = std::bind(&CameraDisplayWidget::consumerThread, this, std::placeholders::_1);
    // cam->setCameraStreamCallback(camera_config.cam_id, handle);
}

CameraDisplayWidget::~CameraDisplayWidget() {
    cleanup();
}

void CameraDisplayWidget::cleanup() {
    av_frame_free(&cpu_frame);
    if (m_render) {
        delete m_render;
        m_render = nullptr;
    }
}

void CameraDisplayWidget::initializeGL() {
    if (m_render) {
        m_render->initialize();
    }
}

void CameraDisplayWidget::paintGL() {
    if (!m_buffer.empty() && m_width > 0 && m_height > 0) {
        m_render->render(m_buffer.data(), m_width, m_height, m_y_stride, m_uv_stride);
    }
}

void CameraDisplayWidget::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
}

void CameraDisplayWidget::consumerThread(Frame frame) {
    AVFrame* src_frame = frame.m_data;
    AVFrame* process_frame = nullptr;

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
    
    if (m_buffer.size() < total_size) {
        m_buffer.resize(total_size);
    }
    memcpy(m_buffer.data(), process_frame->data[0], process_frame->linesize[0] * m_height);
    memcpy(m_buffer.data() + y_size, process_frame->data[1], process_frame->linesize[1] * (m_height / 2));
    av_frame_free(&frame.m_data);
    QMetaObject::invokeMethod(this, "update", Qt::QueuedConnection);
}
