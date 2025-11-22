#ifndef VISIBLE_CAMERA_SHOW_H
#define VISIBLE_CAMERA_SHOW_H

#include <QMainWindow>
#include <QGridLayout>
#include <QLabel>
#include <QWidget>
#include <QThread>
#include <QImage>
#include <vector>
#include <atomic>
#include <mutex>
#include "camera_manager.h"
#include "safe_queue.hpp"
#include "config.h"

extern "C" {
#include <libavutil/frame.h>
}

class CameraDisplayWidget : public QWidget
{
    Q_OBJECT

public:
    explicit CameraDisplayWidget(CameraConfig camera_config, QWidget *parent = nullptr);
    ~CameraDisplayWidget();

private slots:
    void updateFrame(const QImage& image);

private:
    void startVideoThread(); // 从camera_manager获取子码流并显示
    
    int m_cameraIndex;              // 相机编号
    QLabel* m_videoLabel;           // 视频显示标签
    QThread* m_videoThread;          // 视频线程
    std::atomic<bool> m_running;    // 线程运行标志
    
    // 新增：camera_manager相关
    camera_manager* cam;             // 摄像头管理器指针
    safe_queue<Frame>* q;            // 指向子码流队列的指针
    std::mutex m_mutex;              // 保护帧数据的互斥锁
    std::vector<uchar> m_buffer;      // 存储NV12数据的缓冲区
    int m_width, m_height;            // 视频宽高
    int m_y_stride, m_uv_stride;      // Y和UV分量的行步长
    const int m_targetFps = 20;      // 目标帧率（降低到10fps以节省内存）
};

#endif // VISIBLE_CAMERA_SHOW_H