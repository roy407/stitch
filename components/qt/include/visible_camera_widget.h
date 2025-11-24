// visible_camera_widget.h
#ifndef WIDGET_H
#define WIDGET_H

#include <QOpenGLWidget>
#include <QThread>
#include <mutex>
#include <atomic>
#include <vector>
#include "nv12render.h"
#include "camera_manager.h"
#include "safe_queue.hpp"
#include "tools.hpp"
extern "C" {
#include <libavutil/frame.h>
}

class visible_camera_widget : public QOpenGLWidget {
    Q_OBJECT
public:
    explicit visible_camera_widget(QWidget *parent = nullptr);
    ~visible_camera_widget();
signals:
    void VisibleTitle(const QString& title); 
protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;
    void consumerThread();
private slots:
    void VisibleTitleTime(double dec_to_stitch);  
private:
    Nv12Render* m_render;
    camera_manager* cam;
    QThread* con;
    std::mutex m_mutex;
    std::atomic<bool> running;
    safe_queue<Frame>* q;
    std::vector<uchar> m_buffer;
    int m_width;
    int m_height;
    int m_y_stride;
    int m_uv_stride;
    std::chrono::steady_clock::time_point last_title_update;
    std::chrono::seconds update_interval;
    
    void cleanup();
    void* aligned_alloc(size_t size, size_t alignment);
    void aligned_free(void* ptr);
};

#endif // WIDGET_H