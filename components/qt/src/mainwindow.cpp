#include "mainwindow.h"
#include <QApplication>
#include <QScreen>
#include <QGridLayout>
#include <QDebug>
#include <QCloseEvent>
#include "config.h"
#include "infrared_camera_widget.h"
StitchMainWindow::StitchMainWindow(QWidget *parent)
    : QMainWindow(parent),
      cam(nullptr),
      mainWidget(nullptr),
      mainLayout(nullptr),
      infraredStitchLabel(nullptr),
      infraredStitchWidget(nullptr),
      visibleStitchLabel(nullptr),
      visibleStitchWidget(nullptr),
      camerasLabel(nullptr),
      camerasWidget(nullptr),
      camerasLayout(nullptr)
{
    // 统一启动摄像头管理器（只启动一次）
    cam = camera_manager::GetInstance();
    cam->start();
    
    setupUI();
    setupCameras();
}

StitchMainWindow::~StitchMainWindow()
{      
    cameraDisplayWidgets.clear();
    
    // Qt会自动清理所有子对象（包括Widget、Label等）
}

void StitchMainWindow::closeEvent(QCloseEvent *event)
{
    // 停止摄像头管理器（这会停止所有流，导致Widget线程自然退出）
    if (cam) {
        cam->stop();
    }
    
    // 接受关闭事件
    event->accept();
}

void StitchMainWindow::setInfraredStitchWidget(QWidget* widget)
{
    if (!widget) {
        qWarning() << "setInfraredStitchWidget: widget参数为空";
        return;
    }
    
    // 如果已有红外拼接组件，先从布局中移除并删除
    if (infraredStitchWidget) {
        mainLayout->removeWidget(infraredStitchWidget);
        delete infraredStitchWidget;
        infraredStitchWidget = nullptr;
    }
    
    // 设置新的红外拼接组件
    infraredStitchWidget = widget;
    infraredStitchWidget->setParent(this);  // 设置父对象
    infraredStitchWidget->setMinimumHeight(200);  // 设置最小高度
    // 设置大小策略：允许水平和垂直拉伸
    infraredStitchWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    
    // 找到红外拼接标签在布局中的位置，在其后插入新的Widget
    int index = mainLayout->indexOf(infraredStitchLabel);
    if (index >= 0) {
        // 在标签后插入，拉伸比例为1
        mainLayout->insertWidget(index + 1, infraredStitchWidget, 1);
    } else {
        // 如果找不到标签，就添加到末尾
        mainLayout->addWidget(infraredStitchWidget, 1);
    }
    
    // 更新标签文字（如果已经接入，可以改为"红外拼接"）
    // infraredStitchLabel->setText("红外拼接");
    
    qDebug() << "红外拼接显示组件已设置";
}

void StitchMainWindow::setupUI()
{
    // 设置窗口标题
    setWindowTitle("拼接式全景视觉系统设计");
    
    // 创建主窗口中心部件
    mainWidget = new QWidget(this);
    setCentralWidget(mainWidget);
    
    // 创建主布局（垂直布局）
    mainLayout = new QVBoxLayout(mainWidget);
    mainLayout->setSpacing(10);
    mainLayout->setContentsMargins(10, 10, 10, 10);
    
    QFont labelFont;
    labelFont.setPointSize(16);
    labelFont.setBold(true);
    
    // ========== 上层：红外拼接（可通过setInfraredStitchWidget接口设置显示组件） ==========
     infraredStitchLabel = new QLabel("红外拼接", this);
    infraredStitchLabel->setAlignment(Qt::AlignCenter);
    infraredStitchLabel->setFont(labelFont);
    infraredStitchLabel->setStyleSheet("QLabel { background-color: #34495e; color: white; padding: 8px; }");
    mainLayout->addWidget(infraredStitchLabel);
    
    // 检查是否有红外相机配置
    auto& config = config::GetInstance();
    auto IR_cameras = config.GetIRCameraConfig();
    
    // 注意：只有当camera_manager中create_channel_2()被调用时，才创建InfraredWidget
    // 目前create_channel_2()被注释掉了，所以暂时不创建InfraredWidget，避免段错误
    // 如果后续需要启用红外拼接，需要取消注释camera_manager中的create_channel_2()调用
    if (IR_cameras.size() > 0) {
        // 有红外相机，创建红外拼接显示组件
        InfraredWidget* infraredWidget = new InfraredWidget(this);
        infraredStitchWidget = infraredWidget;
        infraredStitchWidget->setMinimumHeight(200);
        // 设置大小策略：允许水平和垂直拉伸
        infraredStitchWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        // 添加时指定拉伸比例：1（红外拼接区域）
        InfraredWidget* irWidget = qobject_cast<InfraredWidget*>(infraredStitchWidget);
        connect(irWidget, &InfraredWidget::IRTitle, 
                this, [this](const QString& title) {
            // 更新红外拼接标签的标题，而不是主窗口标题
            infraredStitchLabel->setText(title);
           
        });
        mainLayout->addWidget(infraredStitchWidget, 1, Qt::AlignCenter);
        
    } else {
        // 没有红外相机或create_channel_2()未调用，创建黑色占位符
        infraredStitchWidget = new QWidget(this);
        infraredStitchWidget->setMinimumHeight(200);
        infraredStitchWidget->setStyleSheet("QWidget { background-color: black; }");
        infraredStitchWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        
        mainLayout->addWidget(infraredStitchWidget, 1);
        infraredStitchLabel->setText("红外拼接");
    }
    
    // ========== 中层：可见光拼接 ==========
    visibleStitchLabel = new QLabel("可见光拼接", this);
    visibleStitchLabel->setAlignment(Qt::AlignCenter);
    visibleStitchLabel->setFont(labelFont);
    visibleStitchLabel->setStyleSheet("QLabel { background-color: #34495e; color: white; padding: 8px; }");
    mainLayout->addWidget(visibleStitchLabel);
    
    // 创建可见光拼接显示组件
    visibleStitchWidget = new visible_camera_widget(this);
    visibleStitchWidget->setMinimumHeight(360);  // 设置最小高度
    // 设置大小策略：允许水平和垂直拉伸（这是主要显示区域，应该占据更多空间）
    visibleStitchWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    // 添加时指定拉伸比例：3（可见光拼接区域，占据更多空间）
    visible_camera_widget* visibleWidget = qobject_cast<visible_camera_widget*>(visibleStitchWidget);
   
    connect(visibleWidget, &visible_camera_widget::VisibleTitle, 
                this, [this](const QString& title) {
           
            visibleStitchLabel->setText(title);
           
        });
   

 
    mainLayout->addWidget(visibleStitchWidget, 3);
    
    // ========== 下层：8路可见光相机 ==========
    camerasLabel = new QLabel("单路可见光相机", this);
    camerasLabel->setAlignment(Qt::AlignCenter);
    camerasLabel->setFont(labelFont);
    camerasLabel->setStyleSheet("QLabel { background-color: #34495e; color: white; padding: 8px; }");
    mainLayout->addWidget(camerasLabel);
    
    // 创建相机容器
    camerasWidget = new QWidget(this);
    camerasLayout = new QGridLayout(camerasWidget);
    camerasLayout->setSpacing(5);
    camerasLayout->setContentsMargins(5, 5, 5, 5);
    // 添加时指定拉伸比例：2（8路相机区域）
    mainLayout->addWidget(camerasWidget, 2);
    setFixedSize(2560, 1440);
}

void StitchMainWindow::setupCameras()
{
    // 从配置文件读取摄像头信息
    auto& config = config::GetInstance();
    auto cameras = config.GetCameraConfig();
    
    int row = 0, col = 0;
    for (size_t i = 0; i < cameras.size(); ++i) {
        if(cameras[i].resize == true) {
            // 创建摄像头显示组件（从camera_manager获取子码流）
            CameraDisplayWidget* cameraWidget = new CameraDisplayWidget(cameras[i], this);
            cameraDisplayWidgets.push_back(cameraWidget);
            
            // 添加到网格布局：2行4列
            camerasLayout->addWidget(cameraWidget, row, col);
        }
        
        col++;
        if (col >= 4) {
            col = 0;
            row++;
        }
    }
}
