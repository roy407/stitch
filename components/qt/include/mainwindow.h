#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QMainWindow>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QStackedLayout>
#include <QWidget>
#include <QLabel>
#include "visible_camera_widget.h"
#include "infrared_camera_widget.h"
#include "visible_single_camera_widget.h"
#include "full_stitch_widget.h"
// #include "camera_manager.h" // [DECOUPLED]

class StitchMainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit StitchMainWindow(QWidget *parent = nullptr);
    ~StitchMainWindow();

 
    void setInfraredStitchWidget(QWidget* widget);

protected:
    void closeEvent(QCloseEvent *event) override;
    void mousePressEvent(QMouseEvent* event) override;

private:
    void setupUI();  // 设置UI界面
    void setupCameras();  // 设置8路可见光相机（子码流）
    
    // 摄像头管理器（统一管理，避免重复启动）
    // camera_manager* cam; // [DECOUPLED]

    QWidget *central{nullptr};
    QStackedLayout* stackedLayout{nullptr};
    
    // UI组件
    QWidget* mainWidget;          // 主窗口中心部件
    QVBoxLayout* mainLayout;      // 主垂直布局
    
    // 上层：红外拼接（黑色占位符，暂未接入信号流）
    QLabel* infraredStitchLabel;  // 红外拼接标题
    QWidget* infraredStitchWidget; // 红外拼接显示组件（黑色占位符）
    
    // 中层：可见光拼接
    QLabel* visibleStitchLabel;   // 可见光拼接标题
    visible_camera_widget* visibleStitchWidget;  // 可见光拼接显示组件
    
    // 下层：8路可见光相机（子码流）
    QLabel* camerasLabel;         // 相机标题
    QWidget* camerasWidget;       // 相机容器
    QGridLayout* camerasLayout;   // 相机网格布局（2行4列）
    std::vector<CameraDisplayWidget*> cameraDisplayWidgets;  // 8个相机显示组件

    full_stitch_widget* full_size_stitch_widget{nullptr};
};

#endif // MAIN_WINDOW_H