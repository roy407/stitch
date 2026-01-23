// detection_widget.h
#ifndef DETECTION_WIDGET_H
#define DETECTION_WIDGET_H

#include <QOpenGLWidget>
#include <QTimer>
#include <QThread>
#include <mutex>
#include <atomic>
#include <vector>
#include <memory>
#include <string>
#include <opencv2/core.hpp>
#include "detection_render.h"
#include "camera_manager.h"
#include "list_queue.hpp"
#include <onnxruntime_cxx_api.h>
// 前向声明
class YoloOnnxDetector;

extern "C" {
#include <libavutil/frame.h>


}

struct DisplayDetection {
    cv::Rect box;
    float confidence;
    std::string label;
};

class detection_widget : public QOpenGLWidget {
    Q_OBJECT

    // 叠加层（纯 QPainter）需要读取检测结果/尺寸信息；避免在 paintGL 里混用 QPainter 导致黑屏
    friend class detection_overlay;

public:
    explicit detection_widget(int pipeline_id, QWidget *parent = nullptr);
    ~detection_widget();

    // 初始化检测器（可选）
    bool initializeDetector(const std::string& model_path, const std::string& labels_path);

signals:
    void DetectionTitle(const QString& title);
    void DetectionToMainWindow();

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;
    


private slots:
    void updateFrame();
    void DetectionTitleTime(double latency, int detection_count);


private:
    detection_render* m_render;
    camera_manager* cam_;
    FrameChannel* stitch_stream_;
    QThread* consumer_thread_;
    std::atomic<bool> running_;
    int pipeline_id_;

    // 检测器
    std::unique_ptr<YoloOnnxDetector> detector_;

    // 帧数据缓冲区
    std::mutex m_mutex;
    std::vector<uchar> m_buffer;
    int m_width;
    int m_height;
    int m_y_stride;
    int m_uv_stride;

    // 检测结果
    std::atomic<int> last_detection_count_;
    std::vector<DisplayDetection> m_detections;

    // UI更新
    QTimer* update_timer_;
    std::chrono::steady_clock::time_point last_title_update;
    std::chrono::seconds update_interval;

    void cleanup();
    void consumerThread(); 
    cv::Mat  nv12ToBGR(const uint8_t* nv12_data, int width, int height, int y_stride, int uv_stride);
    void clampRect(cv::Rect& rect, int img_width, int img_height);
    std::vector<DisplayDetection> detectWithSlidingWindow(const cv::Mat& rgb_image, int window_size, int stride);
};

#endif // DETECTION_WIDGET_H
