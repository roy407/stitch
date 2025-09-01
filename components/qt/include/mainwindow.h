#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include "camera_manager.h"
#include "safe_queue.hpp"
#include "tools.hpp"

extern "C" {
    #include "libavformat/avformat.h"
    #include "libavcodec/avcodec.h"
    #include "libavutil/pixfmt.h"
    #include "libavutil/pixdesc.h"
    #include "libavutil/opt.h"
    #include "libavutil/log.h"
    #include "libavcodec/bsf.h"
    #include <libswscale/swscale.h>   // sws_scale, sws_getContext
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void showFrame(AVFrame* frame); // 接收并显示 AVFrame
private:
    QLabel* videoLabel;
    SwsContext* swsContext = nullptr;     // 复用转换上下文
    AVFrame* cpuFrame = nullptr;          // 复用CPU帧
    AVFrame* rgbFrame = nullptr;          // 复用RGB帧
    uint8_t* rgbBuffer = nullptr;         // 复用RGB缓冲区
    int lastWidth = -1;
    int lastHeight = -1;
    AVPixelFormat lastFormat = AV_PIX_FMT_NONE;
    camera_manager *cam;
    QThread* sf;
    std::atomic<bool> running{true};
};
#endif // MAINWINDOW_H
