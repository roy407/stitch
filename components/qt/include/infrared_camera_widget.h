// infrared_widget.h
#ifndef INFRARED_WIDGET_H
#define INFRARED_WIDGET_H

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

class InfraredWidget : public QOpenGLWidget {
    Q_OBJECT
public:
    explicit InfraredWidget(QWidget *parent = nullptr);
    ~InfraredWidget();
signals:
    void IRTitle(const QString& title); 
protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;
    void consumerThread();

private slots:
    void IRTitleTime(double dec_to_stitch);  
private:
    Nv12Render* m_render;
    camera_manager* cam;
    QThread* con;
    std::mutex m_mutex;
    std::chrono::steady_clock::time_point last_title_update;
    std::chrono::seconds update_interval;
    std::atomic<bool> running;
    FrameChannel* q;
    std::vector<uchar> m_buffer;
    int m_width;
    int m_height;
    int m_y_stride;
    int m_uv_stride;
    
    void cleanup();
};

#endif // INFRARED_WIDGET_H

