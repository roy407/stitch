// widget_for_test.cpp
#include "widget_for_test.h"
#include <QDebug>
#include <QMetaObject>
#include <memory>
#include <QLoggingCategory>
extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
}
#include "log.hpp"

void* widget_for_test::aligned_alloc(size_t size, size_t alignment) {
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

void widget_for_test::aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

widget_for_test::widget_for_test(int pipeline_id, int width, int height, QWidget *parent) : 
    QOpenGLWidget(parent),
    m_render(nullptr),
    m_width(0),
    m_height(0),
    m_y_stride(0),
    m_uv_stride(0)
{
    LOG_INFO("widget test start");
    setFixedSize(width, height);
    QLoggingCategory::setFilterRules("*.debug=false\n*.warning=false");
    m_render = new Nv12Render();
    // cam = camera_manager::GetInstance(); // [DECOUPLED]
    // cam->start();
    cpu_frame = av_frame_alloc();  
    // auto callback_handle = std::bind(&widget_for_test::consumerThread, this, std::placeholders::_1);
    // cam->setStitchStreamCallback(0, callback_handle);

    // [MODIFIED BEGIN] - 初始化SHM接收器
    std::string shm_name = (pipeline_id == 0) ? "/stitch_view_shm" : "/stitch_view_shm_ir";
    m_shm_receiver = new ShmReceiver(shm_name);
    if (m_shm_receiver->init()) {
        qDebug() << "widget_for_test: ShmReceiver initialized for " << QString::fromStdString(shm_name);
        m_running = true;
        m_recv_thread = std::thread(&widget_for_test::shmLoop, this);
    } else {
        qWarning() << "widget_for_test: Failed to init ShmReceiver!";
    }
    // [MODIFIED END]
}

widget_for_test::~widget_for_test() {
    cleanup();
}

void widget_for_test::cleanup() {
    // Stop thread first
    m_running = false;
    if (m_recv_thread.joinable()) {
        m_recv_thread.join();
    }
    if (m_shm_receiver) {
        delete m_shm_receiver;
        m_shm_receiver = nullptr;
    }

    // if (cam) {
    //    cam->stop();
    //    LOG_DEBUG("camera_service stopped!");
    // }
    av_frame_free(&cpu_frame);
    if (m_render) {
        delete m_render;
        m_render = nullptr;
    }
}

void widget_for_test::initializeGL() {
    if (m_render) {
        m_render->initialize();
    }
}

void widget_for_test::paintGL() {
    if (!m_buffer.empty() && m_width > 0 && m_height > 0) {
        m_render->render(m_buffer.data(), m_width, m_height, m_y_stride, m_uv_stride);
    }
}

void widget_for_test::resizeGL(int w, int h) {
    // glViewport(0, 0, w, h);
    if (m_render) {
        // m_render->init(w, h);
    }
}

// [ADDED] Implementation of shmLoop
void widget_for_test::shmLoop() {
    AVFrame* frame = av_frame_alloc();
    
    while (m_running) {
        // 阻塞等待，超时时间 100ms
        if (m_shm_receiver && m_shm_receiver->recvFrame(frame, 100)) {
            // 收到新帧，进行处理
            processFrame(frame);
            
            // 清理引用，准备接收下一帧
            av_frame_unref(frame);
        }
    }
    av_frame_free(&frame);
}

// [ADDED] Implementation of processFrame
void widget_for_test::processFrame(AVFrame* frame) {
    if(!frame) return;
    std::lock_guard<std::mutex> lock(m_mutex);
    
    m_width = frame->width;
    m_height = frame->height;
    m_y_stride = frame->linesize[0];
    m_uv_stride = frame->linesize[1];
    
    size_t y_size = m_y_stride * m_height;
    size_t uv_size = m_uv_stride * (m_height / 2);
    size_t total_size = y_size + uv_size;

    if (m_buffer.size() < total_size) {
        m_buffer.resize(total_size);
    }
    
    // Copy Y Component
    memcpy(m_buffer.data(), frame->data[0], y_size);
    
    // Copy UV Component
    memcpy(m_buffer.data() + y_size, frame->data[1], uv_size);

    QMetaObject::invokeMethod(this, "update", Qt::QueuedConnection);
}

void widget_for_test::consumerThread(Frame frame) {
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
