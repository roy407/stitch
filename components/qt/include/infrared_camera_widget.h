// infrared_widget.h
#ifndef INFRARED_WIDGET_H
#define INFRARED_WIDGET_H

#include <atomic>
#include <chrono>
#include <mutex>
#include <vector>

#include <QOpenGLWidget>
#include <QThread>

extern "C" {
    #include <libavutil/frame.h>
}

#include "camera_manager.h"

#include "nv12render.h"

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
    void consumerThread(Frame frame);

private slots:
    void IRTitleTime(double dec_to_stitch);  
private:
    Nv12Render* m_render;
    camera_manager* cam;
    AVFrame* cpu_frame;
    std::mutex m_mutex;
    std::chrono::steady_clock::time_point last_title_update;
    std::chrono::seconds update_interval;
    std::vector<uchar> m_buffer;
    int m_width;
    int m_height;
    int m_y_stride;
    int m_uv_stride;
    
    void cleanup();
};

#endif // INFRARED_WIDGET_H

