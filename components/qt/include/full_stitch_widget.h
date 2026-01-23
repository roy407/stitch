// full_stitch_widget.h
#ifndef FULL_STITCH_WIDGET_H
#define FULL_STITCH_WIDGET_H

#include <QOpenGLWidget>
#include <QThread>
#include <mutex>
#include <atomic>
#include <vector>
#include "nv12render.h"
#include "camera_manager.h"
#include "list_queue.hpp"

extern "C" {
#include <libavutil/frame.h>
}

class full_stitch_widget : public QOpenGLWidget {
    Q_OBJECT
public:
    explicit full_stitch_widget(QWidget *parent = nullptr);
    ~full_stitch_widget();

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
};

#endif // FULL_STITCH_WIDGET_H