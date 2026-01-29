// visible_camera_widget.cpp
#include "visible_camera_widget.h"
#include <QDebug>
#include <QMetaObject>
#include <memory>
#include <QLoggingCategory>
extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
}
#include "log.hpp"
#include "tools.hpp"

void* visible_camera_widget::aligned_alloc(size_t size, size_t alignment) {
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

void visible_camera_widget::aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}
//可见光拼接初始化
visible_camera_widget::visible_camera_widget(QWidget *parent) : 
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
    // 不再使用setFixedSize，这样Widget可以随父窗口缩放
    
    QLoggingCategory::setFilterRules("*.debug=false\n*.warning=false");
    m_render = new Nv12Render();
    cpu_frame = av_frame_alloc();  
    
    // [MODIFIED BEGIN] - 以前使用回调，现在切换为ShmReceiver
    // auto callback_handle = std::bind(&visible_camera_widget::consumerThread, this, std::placeholders::_1);
    // cam->setStitchStreamCallback(0, callback_handle);
    
    m_shm_receiver = new ShmReceiver("/stitch_view_shm");
    if (m_shm_receiver->init()) {
        qDebug() << "ShmReceiver initialized successfully!";
        m_running = true;
        m_recv_thread = std::thread(&visible_camera_widget::shmLoop, this);
    } else {
        qCritical() << "Failed to init ShmReceiver!";
    }
    // [MODIFIED END]
}
//这边都是原来可见光拼接的
visible_camera_widget::~visible_camera_widget() {
    // [MODIFIED BEGIN] - 停止线程并释放资源
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

void visible_camera_widget::cleanup() {
    av_frame_free(&cpu_frame);
    if (m_render) {
        delete m_render;
        m_render = nullptr;
    }
}

void visible_camera_widget::initializeGL() {
    if (m_render) {
        m_render->initialize();
    }
}

void visible_camera_widget::paintGL() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_buffer.empty() && m_width > 0 && m_height > 0) {
        m_render->render(m_buffer.data(), m_width, m_height, m_y_stride, m_uv_stride);
    }
}
// [MODIFIED BEGIN] - 新增SHM拉流循环线程
void visible_camera_widget::shmLoop() {
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

// 提取原consumerThread中的渲染逻辑
void visible_camera_widget::processFrame(AVFrame* process_frame) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    m_width = process_frame->width;
    m_height = process_frame->height;
    m_y_stride = process_frame->linesize[0];
    m_uv_stride = process_frame->linesize[1];
    
    // 可能还需要画线操作，如果 process_frame 对于画线函数是只读的或者可写的
    // 注意：SHM出来的frame数据是新申请的内存，可以修改
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
    
    // 调整缓冲区大小
    if (m_buffer.size() < total_size) {
        m_buffer.resize(total_size);
    }
    
    // 复制Y和UV数据
    memcpy(m_buffer.data(), process_frame->data[0], process_frame->linesize[0] * m_height);
    memcpy(m_buffer.data() + y_size, process_frame->data[1], process_frame->linesize[1] * (m_height / 2));
    
    // 触发界面刷新
    QMetaObject::invokeMethod(this, "update", Qt::QueuedConnection);
}
// [MODIFIED END] -- 原 consumerThread 可保留作为兼容或者删除

void visible_camera_widget::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
}
void visible_camera_widget::VisibleTitleTime(double cost_time){
    QString title = QString("可见光拼接");
    emit VisibleTitle(title);
} 

// [MODIFIED] - 旧的回调函数，已废弃。保留空壳防止链接错误，但移除实现。
void visible_camera_widget::consumerThread(Frame frame) {  
    // 释放可能传入的AVFrame，防止内存泄漏，但不再进行渲染
    if (frame.m_data) {
        av_frame_free(&frame.m_data);
    }
    // qWarning() << "Old consumerThread called unexpectedly! Ignoring frame.";
}
