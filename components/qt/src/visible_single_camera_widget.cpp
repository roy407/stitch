#include "visible_single_camera_widget.h"
#include <QVBoxLayout>
#include <QCloseEvent>
#include <QScreen>
#include <QApplication>
#include <QGridLayout>
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

// 辅助函数：将NV12数据转换为QImage（直接缩放到640x360以节省内存）
static QImage convertNV12ToQImage(const uchar* nv12_data, int width, int height, 
                                   int y_stride, int uv_stride) {
    // 目标显示尺寸（固定为640x360以节省内存）
    const int target_width = 640;
    const int target_height = 360;
    
    SwsContext* sws_ctx = sws_getContext(
        width, height, AV_PIX_FMT_NV12,
        target_width, target_height, AV_PIX_FMT_RGB24,  // 直接缩放到目标尺寸
        SWS_BILINEAR, nullptr, nullptr, nullptr
    );
    
    if (!sws_ctx) {
        qWarning() << "Failed to create SwsContext";
        return QImage();
    }
    
    // 分配RGB缓冲区（使用目标尺寸，而不是原始尺寸，节省内存）
    int rgb_linesize = target_width * 3;
    std::vector<uchar> rgb_buffer(target_width * target_height * 3);
    uint8_t* dst_data[1] = { rgb_buffer.data() };
    int dst_linesize[1] = { rgb_linesize };
    
    // 准备源数据指针
    const uint8_t* src_data[2] = {
        nv12_data,                    // Y平面
        nv12_data + y_stride * height // UV平面
    };
    int src_linesize[2] = { y_stride, uv_stride };
    
    // 执行转换和缩放（一步完成，节省内存）
    sws_scale(sws_ctx, src_data, src_linesize, 0, height, 
              dst_data, dst_linesize);
    
    sws_freeContext(sws_ctx);
    
    // 创建QImage
    // 注意：FFmpeg的AV_PIX_FMT_RGB24是RGB格式（R-G-B顺序），Qt的Format_RGB888也是RGB格式
    // sws_scale从NV12转换到RGB24时，输出的是RGB格式，所以不需要rgbSwapped()
    QImage image(rgb_buffer.data(), target_width, target_height, rgb_linesize, QImage::Format_RGB888);
    // 必须copy()，因为rgb_buffer是局部变量，会被销毁
    return image.copy();
}

// CameraDisplayWidget 实现
CameraDisplayWidget::CameraDisplayWidget(int cameraIndex, QWidget *parent)
    : QWidget(parent),
      m_cameraIndex(cameraIndex),
      m_videoLabel(nullptr),
      m_videoThread(nullptr),
      m_running(true),
      cam(nullptr),
      q(nullptr),
      m_width(0),
      m_height(0),
      m_y_stride(0),
      m_uv_stride(0)
{
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setContentsMargins(2, 2, 2, 2);
    
    // 创建标签显示摄像头编号
    QLabel* titleLabel = new QLabel(QString("摄像头 %1").arg(cameraIndex), this);
    titleLabel->setAlignment(Qt::AlignCenter);
    titleLabel->setStyleSheet("QLabel { background-color: #34495e; color: white; padding: 5px; font-weight: bold; }");
    layout->addWidget(titleLabel);
    
    // 创建视频显示标签
    m_videoLabel = new QLabel(this);
    m_videoLabel->setAlignment(Qt::AlignCenter);
    m_videoLabel->setStyleSheet("background-color: black;");
    m_videoLabel->setFixedSize(640, 360);
    m_videoLabel->setScaledContents(true);
    layout->addWidget(m_videoLabel);
    
    // 获取camera_manager实例
    cam = camera_manager::GetInstance();
    
    // 获取对应相机的子码流队列
    // 注意：如果m_sub_stream未初始化，get_single_camera_sub_stream会崩溃
    // 暂时使用拼接流作为替代（虽然会被StitchConsumer消费，但至少不会崩溃）
    // 需要camera_manager正确初始化m_sub_stream后才能使用子码流
     // 获取对应相机的子码流队列（从独立的子码流producer获取）
     q = &(cam->get_single_camera_sub_stream(m_cameraIndex));
    
    startVideoThread();
}

CameraDisplayWidget::~CameraDisplayWidget()
{
    m_running.store(false);
    if (m_videoThread) {
        m_videoThread->quit();
        m_videoThread->wait(1000);
        if (m_videoThread->isRunning()) {
            m_videoThread->terminate();
            m_videoThread->wait();
        }
        delete m_videoThread;
    }
}

void CameraDisplayWidget::startVideoThread()
{
    m_videoThread = QThread::create([this]() {
        AVFrame* cpu_frame = av_frame_alloc();
        if (!cpu_frame) {
            qWarning() << "Failed to allocate CPU frame";
            return;
        }
        
        while (m_running.load()) {
            Frame frame;
            if (!q->wait_and_pop(frame)) { break;
            }
            
            std::unique_lock<std::mutex> lock(m_mutex);
            
            AVFrame* src_frame = frame.m_data;
            AVFrame* process_frame = src_frame;
            
            // 硬件帧转换到CPU（如果需要）
            if (src_frame->format == AV_PIX_FMT_CUDA) {
                if (av_hwframe_transfer_data(cpu_frame, src_frame, 0) < 0) {
                    qWarning() << "Failed to transfer frame to CPU";
                    av_frame_free(&frame.m_data);
                    continue;
                }
                process_frame = cpu_frame;
            }
            
            // 获取帧信息
            m_width = process_frame->width;
            m_height = process_frame->height;
            m_y_stride = process_frame->linesize[0];
            m_uv_stride = process_frame->linesize[1];
            
            // 安全检查：限制最大分辨率，防止内存溢出
            const int MAX_WIDTH = 1920;
            const int MAX_HEIGHT = 1080;
            if (m_width > MAX_WIDTH || m_height > MAX_HEIGHT) {
                qWarning() << "Frame size too large:" << m_width << "x" << m_height 
                          << ", skipping frame to prevent memory overflow";
                av_frame_free(&frame.m_data);
                continue;
            }
            
            // 确保行对齐是32字节的倍数
            if (m_y_stride % 32 != 0) {
                m_y_stride = ((m_y_stride + 31) / 32) * 32;
            }
            if (m_uv_stride % 32 != 0) {
                m_uv_stride = ((m_uv_stride + 31) / 32) * 32;
            }
            
            // 分配缓冲区
            size_t y_size = m_y_stride * m_height;
            size_t uv_size = m_uv_stride * (m_height / 2);
            size_t total_size = y_size + uv_size;
            
            // 限制缓冲区大小，防止内存溢出
            const size_t MAX_BUFFER_SIZE = 1920 * 1080 * 2;  // 最大约4MB
            if (total_size > MAX_BUFFER_SIZE) {
                qWarning() << "Buffer size too large:" << total_size 
                          << "bytes, skipping frame";
                av_frame_free(&frame.m_data);
                continue;
            }
            
            if (m_buffer.size() < total_size) {
                m_buffer.resize(total_size);
            }
            
            // 拷贝Y和UV数据
            memcpy(m_buffer.data(), process_frame->data[0], y_size);
            memcpy(m_buffer.data() + y_size, process_frame->data[1], uv_size);
            
            // 转换为QImage（convertNV12ToQImage已经缩放到640x360，无需再次缩放）
            QImage image = convertNV12ToQImage(m_buffer.data(), m_width, m_height, 
                                               m_y_stride, m_uv_stride);
            
            // 检查图像是否有效（避免发送空图像）
            if (!image.isNull()) {
                // 通过信号槽更新UI（线程安全）
                QMetaObject::invokeMethod(this, "updateFrame", Qt::QueuedConnection,
                                        Q_ARG(QImage, image));
            }
            
            av_frame_free(&frame.m_data);
            lock.unlock();
        }
        
        av_frame_free(&cpu_frame);
    });
    
    m_videoThread->start();
}

void CameraDisplayWidget::updateFrame(const QImage& image)
{
    if (m_videoLabel && !image.isNull()) {
        m_videoLabel->setPixmap(QPixmap::fromImage(image));
    }
}

// VisibleCameraShow 实现
VisibleCameraShow::VisibleCameraShow(QWidget *parent)
    : QMainWindow(parent),
      m_centralWidget(nullptr),
      m_gridLayout(nullptr)
{
    setWindowTitle("可见光摄像头");
    setupUI();
    setupCameras();
}

VisibleCameraShow::~VisibleCameraShow()
{
    for (auto* widget : m_cameraWidgets) {
        delete widget;
    }
}

void VisibleCameraShow::setupUI()
{
    m_centralWidget = new QWidget(this);
    setCentralWidget(m_centralWidget);
    
    m_gridLayout = new QGridLayout(m_centralWidget);
    m_gridLayout->setSpacing(5);
    m_gridLayout->setContentsMargins(10, 10, 10, 10);
    
    // 设置窗口大小：2行4列，每个640x360，加上间距和标题
    resize(640 * 4 + 50, 360 * 2 + 100);
    
    // 居中显示窗口
    QScreen *screen = QApplication::primaryScreen();
    if (screen) {
        QRect screenGeometry = screen->geometry();
        int x = (screenGeometry.width() - width()) / 2;
        int y = (screenGeometry.height() - height()) / 2;
        move(x, y);
    }
}

void VisibleCameraShow::setupCameras()
{
    // 从配置文件读取摄像头数量
    auto& config = config::GetInstance();
    auto cameras = config.GetCameraConfig();
    
    int row = 0, col = 0;
    for (size_t i = 0; i < cameras.size() && i < 8; ++i) {
        // 创建摄像头显示组件（不再需要RTSP URL）
        CameraDisplayWidget* cameraWidget = new CameraDisplayWidget(i, this);
        m_cameraWidgets.push_back(cameraWidget);
        
        // 添加到网格布局：2行4列
        m_gridLayout->addWidget(cameraWidget, row, col);
        
        col++;
        if (col >= 4) {
            col = 0;
            row++;
        }
    }
}

void VisibleCameraShow::closeEvent(QCloseEvent *event)
{
    event->accept();
}