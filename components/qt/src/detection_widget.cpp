// detection_widget.cpp
#include "detection_widget.h"
#include "detection_render.h"
#include "detector.h"
#include <QDebug>
#include <QMetaObject>
#include <QPainter>
#include <QScrollArea>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <chrono>
#include "log.hpp"
#include "tools.hpp"

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
}

detection_widget::detection_widget(int pipeline_id, QWidget *parent)
    : QOpenGLWidget(parent),
      m_render(nullptr),
      cam_(nullptr),
      stitch_stream_(nullptr),
      consumer_thread_(nullptr),
      running_(false),
      pipeline_id_(pipeline_id),
      m_width(0),
      m_height(0),
      m_y_stride(0),
      m_uv_stride(0),
      last_detection_count_(0),
      update_timer_(nullptr),
      last_title_update(std::chrono::steady_clock::now()),
      update_interval(std::chrono::seconds(1))
{
    setFixedSize(20803, 2160);
    
    m_render = new detection_render();
    cam_ = camera_manager::GetInstance();
    
    // 启动消费者线程
    consumer_thread_ = QThread::create([this]() { consumerThread(); });
    consumer_thread_->start();
    
    // 启动定时器更新画面
    update_timer_ = new QTimer(this);
    connect(update_timer_, &QTimer::timeout, this, &detection_widget::updateFrame);
    update_timer_->start(50); // 约20fps
}

detection_widget::~detection_widget() {
    cleanup();
}

void detection_widget::cleanup() {
    running_.store(false);
    
    if (update_timer_) {
        update_timer_->stop();
        update_timer_->deleteLater();
        update_timer_ = nullptr;
    }
    
    if (consumer_thread_) {
        if (stitch_stream_) {
            stitch_stream_->stop();
        }
        consumer_thread_->wait();
        delete consumer_thread_;
        consumer_thread_ = nullptr;
     
    }
    
    if (m_render) {
        delete m_render;
        m_render = nullptr;
    }
}

bool detection_widget::initializeDetector(const std::string& model_path, const std::string& labels_path) {
    try {
        detector_ = std::make_unique<YoloOnnxDetector>(model_path, labels_path);
        if (detector_ && detector_->is_initialized()) {
            LOG_INFO("Detector initialized successfully: {}", model_path);
            emit DetectionTitle("目标检测: 检测器已加载");
            return true;
        } else {
            LOG_ERROR("Detector initialization failed: {}", model_path);
            emit DetectionTitle("目标检测: 检测器加载失败");
            return false;
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Exception during detector initialization: {}", e.what());
        emit DetectionTitle("目标检测: 检测器加载异常");
        return false;
    }
}

void detection_widget::initializeGL() {
    if (m_render) {
        m_render->initialize();
    }
}

void detection_widget::paintGL() {
    const uint8_t* y_ptr = nullptr;
    int width = 0;
    int height = 0;
    int y_stride = 0;
    int uv_stride = 0;
    std::vector<DisplayDetection> detections_copy;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_buffer.empty() && m_width > 0 && m_height > 0) {
            y_ptr = m_buffer.data();
            width = m_width;
            height = m_height;
            y_stride = m_y_stride;
            uv_stride = m_uv_stride;
            detections_copy = m_detections;
        }
    }
    
    if (y_ptr && width > 0 && height > 0 && m_render) {
        // 只做 OpenGL 渲染（避免混用 QPainter 导致黑屏）
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        

        m_render->render(const_cast<uchar*>(y_ptr), width, height, y_stride, uv_stride);

        if(!detections_copy.empty()) {
            std::vector<detection_box> boxes;
            for (const auto& det : detections_copy) {
               detection_box box = {
                    static_cast<float>(det.box.x),      // 显式转换为float
                    static_cast<float>(det.box.y),      // 显式转换为float
                    static_cast<float>(det.box.width),  // 显式转换为float
                    static_cast<float>(det.box.height), // 显式转换为float
                    det.confidence,
                    det.label
                };
            boxes.push_back(box);
        }
        m_render->renderDetections(boxes, width, height);
    }
} else {
        // 没有数据，显示黑屏
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }

}
void detection_widget::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
}

void detection_widget::updateFrame() {
    update(); // 触发 paintGL
}

void detection_widget::DetectionTitleTime(double latency, int detection_count) {
    auto now = std::chrono::steady_clock::now();
    if (now - last_title_update >= update_interval) {
        last_title_update = now;
        QString title = QString("目标检测: %1 个目标, 延迟: %.1fms")
                            .arg(detection_count)
                            .arg(latency);
        emit DetectionTitle(title);
    }
}

void detection_widget::consumerThread() {
    running_.store(true);
    
    // 获取拼接流
    stitch_stream_ = cam_->getStitchCameraStream(0);
    
    
    AVFrame* cpu_frame = av_frame_alloc();
    
    while (running_.load()) {
        Frame frame;
        if (!stitch_stream_->recv(frame)) {
            break;
        }
        
        AVFrame* src_frame = frame.m_data;
        if (!src_frame) {
            continue;
        }
        
        AVFrame* process_frame = src_frame;
        
        // 硬件帧转换到CPU
        if (src_frame->format == AV_PIX_FMT_CUDA) {
            if (av_hwframe_transfer_data(cpu_frame, src_frame, 0) < 0) {
                LOG_WARN("Failed to transfer frame to CPU");
                av_frame_free(&frame.m_data);
                continue;
            }
            process_frame = cpu_frame;
        }
        
        int frame_width = process_frame->width;
        int frame_height = process_frame->height;
        int frame_y_stride = process_frame->linesize[0];
        int frame_uv_stride = process_frame->linesize[1];
        
        // 确保行对齐是32字节的倍数（用于OpenGL渲染）
        int m_y_stride_new = frame_y_stride;
        int m_uv_stride_new = frame_uv_stride;
        
        if (m_y_stride_new % 32 != 0) {
            m_y_stride_new = ((m_y_stride_new + 31) / 32) * 32;
        }
        if (m_uv_stride_new % 32 != 0) {
            m_uv_stride_new = ((m_uv_stride_new + 31) / 32) * 32;
        }
        
        size_t y_size = m_y_stride_new * frame_height;
        size_t uv_size = m_uv_stride_new * (frame_height / 2);
        size_t total_size = y_size + uv_size;
        
        // 更新缓冲区
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            
            m_width = frame_width;
            m_height = frame_height;
            m_y_stride = m_y_stride_new;
            m_uv_stride = m_uv_stride_new;
            
            if (m_buffer.size() < total_size) {
                m_buffer.resize(total_size);
            }
            
            // 复制Y平面
            if (frame_y_stride == m_y_stride_new) {
                // 步长相同，直接复制
                memcpy(m_buffer.data(), process_frame->data[0], frame_y_stride * frame_height);
            } else {
                // 步长不同，逐行复制并填充对齐部分
                for (int y = 0; y < frame_height; ++y) {
                    memcpy(m_buffer.data() + y * m_y_stride_new,
                           process_frame->data[0] + y * frame_y_stride,
                           frame_width);
                    // 填充对齐部分为0
                    if (m_y_stride_new > frame_width) {
                        memset(m_buffer.data() + y * m_y_stride_new + frame_width, 
                               0, m_y_stride_new - frame_width);
                    }
                }
            }
            
            // 复制UV平面
            if (frame_uv_stride == m_uv_stride_new) {
                // 步长相同，直接复制
                memcpy(m_buffer.data() + y_size, process_frame->data[1], 
                       frame_uv_stride * (frame_height / 2));
            } else {
                // 步长不同，逐行复制并填充对齐部分
                for (int y = 0; y < frame_height / 2; ++y) {
                    memcpy(m_buffer.data() + y_size + y * m_uv_stride_new,
                           process_frame->data[1] + y * frame_uv_stride,
                           frame_width);
                    // 填充对齐部分为0
                    if (m_uv_stride_new > frame_width) {
                        memset(m_buffer.data() + y_size + y * m_uv_stride_new + frame_width,
                               0, m_uv_stride_new - frame_width);
                    }
                }
            }
            
            // 执行检测（如果有检测器）
            if (detector_ && detector_->is_initialized()) {
                auto detection_start = std::chrono::steady_clock::now();
                
                // 转换为BGR进行检测
                cv::Mat bgr_image = nv12ToBGR(m_buffer.data(), frame_width, frame_height, 
                                              m_y_stride_new, m_uv_stride_new);
                
                if (!bgr_image.empty()) {
                    // 使用滑动窗口检测
                    m_detections = detectWithSlidingWindow(bgr_image, 640, 320);
                    last_detection_count_.store(static_cast<int>(m_detections.size()));
                    
                    auto detection_end = std::chrono::steady_clock::now();
                    double detection_latency = std::chrono::duration<double, std::milli>(
                        detection_end - detection_start).count();
                    
                    QMetaObject::invokeMethod(this, "DetectionTitleTime", Qt::QueuedConnection,
                                              Q_ARG(double, detection_latency),
                                              Q_ARG(int, static_cast<int>(m_detections.size())));
                }
            }
        }
        
        av_frame_free(&frame.m_data);
        
        // 触发画面更新
        QMetaObject::invokeMethod(this, "update", Qt::QueuedConnection);
    }
    
    if (stitch_stream_) {
        stitch_stream_->clear();
    }
    av_frame_free(&cpu_frame);
    
    LOG_DEBUG("detection_widget consumer thread exited");
}

cv::Mat detection_widget::nv12ToBGR(const uint8_t* nv12_data, int width, int height, 
                                     int y_stride, int uv_stride) {
    if (!nv12_data || width <= 0 || height <= 0 || y_stride <= 0 || uv_stride <= 0) {
        return cv::Mat();
    }
    
    // 确保宽高是偶数
    int even_width = (width / 2) * 2;
    int even_height = (height / 2) * 2;
    
    if (even_width <= 0 || even_height <= 0) {
        return cv::Mat();
    }
    
    // 创建Y和UV平面
    const uint8_t* y_plane = nv12_data;
    const uint8_t* uv_plane = nv12_data + y_stride * height;
    
    // 创建连续的Mat
    cv::Mat y_mat(height, width, CV_8UC1);
    cv::Mat uv_mat(height / 2, width / 2, CV_8UC2);  // 注意：UV平面的宽度是Y的一半！
    
    // 复制Y平面
    for (int y = 0; y < height; ++y) {
        memcpy(y_mat.ptr<uint8_t>(y), y_plane + y * y_stride, width);
    }
    
    // 复制UV平面 - 注意UV的宽度是Y的一半
    int uv_width = width / 2;
    for (int y = 0; y < height / 2; ++y) {
        memcpy(uv_mat.ptr<uint8_t>(y), uv_plane + y * uv_stride, uv_width * 2);  // *2 因为每个UV像素是2个字节
    }
    
    // 使用OpenCV的cvtColorTwoPlane进行NV12到BGR转换
    cv::Mat bgr_image;
    try {
        cv::cvtColorTwoPlane(y_mat(cv::Rect(0, 0, even_width, even_height)),
                            uv_mat(cv::Rect(0, 0, even_width / 2, even_height / 2)),  // UV尺寸是Y的一半
                            bgr_image, cv::COLOR_YUV2BGR_NV12);
    } catch (const cv::Exception& e) {
        LOG_ERROR("OpenCV error in cvtColorTwoPlane: {}", e.what());
        LOG_ERROR("Y size: {}x{}, UV size: {}x{}", 
                  even_width, even_height, even_width/2, even_height/2);
        return cv::Mat();
    }
    
    return bgr_image;
}
void detection_widget::clampRect(cv::Rect& rect, int img_width, int img_height) {
    rect.x = std::max(0, std::min(rect.x, img_width - 1));
    rect.y = std::max(0, std::min(rect.y, img_height - 1));
    int x2 = std::max(0, std::min(rect.x + rect.width, img_width));
    int y2 = std::max(0, std::min(rect.y + rect.height, img_height));
    rect.width = std::max(0, x2 - rect.x);
    rect.height = std::max(0, y2 - rect.y);
}

std::vector<DisplayDetection> detection_widget::detectWithSlidingWindow(
    const cv::Mat& rgb_image, int window_size, int stride) {
    
    std::vector<DisplayDetection> all_detections;
    
    if (!detector_ || !detector_->is_initialized() || rgb_image.empty()) {
        return all_detections;
    }
    
    int img_width = rgb_image.cols;
    int img_height = rgb_image.rows;
    
    if (img_width < window_size || img_height < window_size) {
        // 图像太小，直接检测
        std::vector<DetectionResult> results = detector_->detect(rgb_image);
        const auto& labels = detector_->labels();
        
        for (const auto& result : results) {
            DisplayDetection det;
            det.box = result.box;
            det.confidence = result.confidence;
            if (result.class_id >= 0 && result.class_id < static_cast<int>(labels.size())) {
                det.label = labels[result.class_id];
            } else {
                det.label = "unknown";
            }
            all_detections.push_back(det);
        }
        return all_detections;
    }
    
    // 滑动窗口检测
    for (int y = 0; y <= img_height - window_size; y += stride) {
        for (int x = 0; x <= img_width - window_size; x += stride) {
            cv::Rect window_rect(x, y, window_size, window_size);
            
            // 提取窗口区域
            cv::Mat window = rgb_image(window_rect);
            
            // 检测
            std::vector<DetectionResult> results = detector_->detect(window);
            const auto& labels = detector_->labels();
            
            // 转换检测框坐标到原图坐标系
            for (const auto& result : results) {
                DisplayDetection det;
                det.box = cv::Rect(result.box.x + x, 
                                   result.box.y + y,
                                   result.box.width,
                                   result.box.height);
                
                // 限制在图像范围内
                clampRect(det.box, img_width, img_height);
                
                if (det.box.area() > 0) {
                    det.confidence = result.confidence;
                    if (result.class_id >= 0 && result.class_id < static_cast<int>(labels.size())) {
                        det.label = labels[result.class_id];
                    } else {
                        det.label = "unknown";
                    }
                    all_detections.push_back(det);
                }
            }
        }
    }
    
    // 非极大值抑制（NMS）去重
    if (all_detections.size() > 1) {
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        
        for (const auto& det : all_detections) {
            boxes.push_back(det.box);
            confidences.push_back(det.confidence);
        }
        
        std::vector<int> kept;
        cv::dnn::NMSBoxes(boxes, confidences, 0.25f, 0.45f, kept);
        
        std::vector<DisplayDetection> filtered_detections;
        for (int idx : kept) {
            filtered_detections.push_back(all_detections[idx]);
        }
        
        return filtered_detections;
    }
    
    return all_detections;
}



