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
    setMinimumSize(640, 360);  // 最小尺寸：360p
    // 不再使用setFixedSize，这样Widget可以随父窗口缩放
    
    QLoggingCategory::setFilterRules("*.debug=false\n*.warning=false");
    m_render = new Nv12Render();
    // 只获取实例，不启动（由主窗口统一管理）
    cam = camera_manager::GetInstance();
   q = cam->getStitchCameraStream(0);
    con = QThread::create([this](){consumerThread();});
    con->start();
}
//这边都是原来可见光拼接的
visible_camera_widget::~visible_camera_widget() {
    cleanup();
}

void visible_camera_widget::cleanup() {
    running.store(false);
    if (con) {
        q->stop();
        con->wait();
        delete con;
        con = nullptr;
        LOG_DEBUG("visible_camera_widget consumer thread destroyed!");
    }
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

void visible_camera_widget::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
}
void visible_camera_widget::VisibleTitleTime(double cost_time){
    QString title = QString("可见光拼接");
    emit VisibleTitle(title);
} 

        
//可见光拼接帧的处理
void visible_camera_widget::consumerThread() {
    static std::string filename = std::string("build/") + get_current_time_filename(".csv");

    std::ofstream ofs(filename, std::ios::app);  // 追加写入
    if (!ofs.is_open()) {
        LOG_ERROR("Failed to open file: {}", filename);
        return;
    }

    AVFrame* cpu_frame = av_frame_alloc();    
    while (running.load()) {
        Frame frame;
  
        if(!q->recv(frame)) break;
        std::unique_lock<std::mutex> lock(m_mutex, std::try_to_lock);
        double dec_to_stitch = 0.0;
        int active_cam_count = 0;
        frame.m_costTimes.when_show_on_the_screen = get_now_time();
        for (int i = 0; i < 10; ++i) {
            if (frame.m_costTimes.when_get_packet[i] != 0) {
                double cam_dec_to_stitch = (frame.m_costTimes.when_show_on_the_screen - frame.m_costTimes.when_get_packet[i]) * 1e-6;
                dec_to_stitch += cam_dec_to_stitch;
                active_cam_count++;
            }
        }
        
        if (active_cam_count > 0) {
            dec_to_stitch = dec_to_stitch / active_cam_count;
        }
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
        
        av_frame_free(&frame.m_data);
        
        auto now = std::chrono::steady_clock::now();
        if (now - last_title_update >= update_interval) {
       
        QMetaObject::invokeMethod(this, "VisibleTitleTime", Qt::QueuedConnection, 
                                Q_ARG(double, dec_to_stitch));
              last_title_update = now;  // 更新时间戳
        }
        QMetaObject::invokeMethod(this, "update", Qt::QueuedConnection);
        save_cost_table_csv(frame.m_costTimes,ofs);
    }
    q->clear();
    av_frame_free(&cpu_frame);
}
