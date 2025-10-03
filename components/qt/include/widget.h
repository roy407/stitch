// widget.h
#ifndef WIDGET_H
#define WIDGET_H

#include <QOpenGLWidget>
#include <QThread>
#include <mutex>
#include <atomic>
#include <vector>
#include "nv12render.h"
#include "camera_manager.h"

extern "C" {
#include <libavutil/frame.h>
}

class Widget : public QOpenGLWidget {
    Q_OBJECT
public:
    explicit Widget(QWidget *parent = nullptr);
    ~Widget();

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;

private:
    Nv12Render* m_render;
    camera_manager* cam;
    QThread* sf;
    std::mutex m_mutex;
    std::atomic<bool> running;
    
    std::vector<uchar> m_buffer;
    int m_width;
    int m_height;
    int m_y_stride;
    int m_uv_stride;
    
    void cleanup();
    void* aligned_alloc(size_t size, size_t alignment);
    void aligned_free(void* ptr);
};

#endif // WIDGET_H