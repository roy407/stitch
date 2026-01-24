// full_stitch_widget.h
#ifndef FULL_STITCH_WIDGET_H
#define FULL_STITCH_WIDGET_H

#include <atomic>
#include <mutex>
#include <vector>

#include <QOpenGLWidget>
#include <QThread>

extern "C" {
    #include <libavutil/frame.h>
}

#include "camera_manager.h"

#include "nv12render.h"

class full_stitch_widget : public QOpenGLWidget {
    Q_OBJECT
public:
    explicit full_stitch_widget(QWidget *parent = nullptr);
    ~full_stitch_widget();

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;
    void consumerThread(Frame frame);

private:
    AVFrame* cpu_frame;
    Nv12Render* m_render;
    camera_manager* cam;
    std::mutex m_mutex;
    std::vector<uchar> m_buffer;
    int m_width;
    int m_height;
    int m_y_stride;
    int m_uv_stride;
    
    void cleanup();
    void* aligned_alloc(size_t size, size_t alignment);
    void aligned_free(void* ptr);
};

#endif // FULL_STITCH_WIDGET_H