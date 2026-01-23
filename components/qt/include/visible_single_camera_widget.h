#ifndef VISIBLE_CAMERA_SHOW_H
#define VISIBLE_CAMERA_SHOW_H

#include <QGridLayout>
#include <QOpenGLWidget>
#include <QLabel>
#include <QWidget>
#include <QThread>
#include <QImage>
#include <vector>
#include <atomic>
#include <mutex>
#include "camera_manager.h"
#include "list_queue.hpp"
#include "config.h"
#include "nv12render.h"

extern "C" {
#include <libavutil/frame.h>
}

class CameraDisplayWidget : public QOpenGLWidget {
    Q_OBJECT

public:
    explicit CameraDisplayWidget(CameraConfig camera_config, QWidget *parent = nullptr);
    ~CameraDisplayWidget();

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;
    void consumerThread();

private:
    Nv12Render* m_render;
    
    camera_manager* cam;
    QThread* con;
    std::mutex m_mutex;
    std::atomic<bool> running;
    FrameChannel* q;
    std::vector<uchar> m_buffer;
    int m_width;
    int m_height;
    int m_y_stride;
    int m_uv_stride;
    
    void cleanup();
    void* aligned_alloc(size_t size, size_t alignment);
    void aligned_free(void* ptr);
};

#endif // VISIBLE_CAMERA_SHOW_H