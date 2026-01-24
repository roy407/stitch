// widget_for_test.h
#ifndef WIDGET_FOR_TEST_H
#define WIDGET_FOR_TEST_H

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

class widget_for_test : public QOpenGLWidget {
    Q_OBJECT
public:
    explicit widget_for_test(int pipeline_id, int width, int height, QWidget *parent = nullptr);
    ~widget_for_test();

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;
    void consumerThread(Frame frame);

private:
    Nv12Render* m_render;
    camera_manager* cam;
    AVFrame* cpu_frame;
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

#endif // WIDGET_FOR_TEST_H