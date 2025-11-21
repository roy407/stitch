#include "mainwindow.h"
#include <QApplication>
#include <QScreen>
#include <QGridLayout>
#include <QDebug>
#include <QCloseEvent>
#include "config.h"

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
    // 停止摄像头管理器
    if (cam) {
        cam->stop();
    }
    
    
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
        mainLayout->addWidget(infraredStitchWidget, 1, Qt::AlignCenter);
    } else {
        // 没有红外相机或create_channel_2()未调用，创建黑色占位符
        infraredStitchWidget = new QWidget(this);
        infraredStitchWidget->setMinimumHeight(200);
        infraredStitchWidget->setStyleSheet("QWidget { background-color: black; }");
        infraredStitchWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        mainLayout->addWidget(infraredStitchWidget, 1);
        infraredStitchLabel->setText("红外拼接（暂未接入）");
    }
    
    // ========== 中层：可见光拼接 ==========
    visibleStitchLabel = new QLabel("可见光拼接", this);
    visibleStitchLabel->setAlignment(Qt::AlignCenter);
    visibleStitchLabel->setFont(labelFont);
    visibleStitchLabel->setStyleSheet("QLabel { background-color: #34495e; color: white; padding: 8px; }");
    mainLayout->addWidget(visibleStitchLabel);
    
    // 创建可见光拼接显示组件
    visibleStitchWidget = new Widget(this);
    visibleStitchWidget->setMinimumHeight(360);  // 设置最小高度
    // 设置大小策略：允许水平和垂直拉伸（这是主要显示区域，应该占据更多空间）
    visibleStitchWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    // 添加时指定拉伸比例：3（可见光拼接区域，占据更多空间）
    mainLayout->addWidget(visibleStitchWidget, 3);
    
    // ========== 下层：8路可见光相机（子码流） ==========
    camerasLabel = new QLabel("8路可见光相机（子码流）", this);
    camerasLabel->setAlignment(Qt::AlignCenter);
    camerasLabel->setFont(labelFont);
    camerasLabel->setStyleSheet("QLabel { background-color: #34495e; color: white; padding: 8px; }");
    mainLayout->addWidget(camerasLabel);
    
    // 创建相机容器
    camerasWidget = new QWidget(this);
    camerasLayout = new QGridLayout(camerasWidget);
    camerasLayout->setSpacing(5);
    camerasLayout->setContentsMargins(5, 5, 5, 5);
    // 设置大小策略：允许水平和垂直拉伸
    camerasWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    // 添加时指定拉伸比例：2（8路相机区域）
    mainLayout->addWidget(camerasWidget, 2);
    
    // 设置窗口最小尺寸（允许用户缩放窗口）
    setMinimumSize(1280, 720);  // 最小尺寸：720p
    resize(1920, 1080);  // 初始尺寸：1080p
    
    // 居中显示窗口
    QScreen *screen = QApplication::primaryScreen();
    if (screen) {
        QRect screenGeometry = screen->geometry();
        int x = (screenGeometry.width() - width()) / 2;
        int y = (screenGeometry.height() - height()) / 2;
        move(x, y);
    }
}

void StitchMainWindow::setupCameras()
{
    // 从配置文件读取摄像头信息
    auto& config = config::GetInstance();
    auto cameras = config.GetCameraConfig();
    
    int row = 0, col = 0;
    for (size_t i = 0; i < cameras.size() && i < 8; ++i) {
        // 创建摄像头显示组件（从camera_manager获取子码流）
        CameraDisplayWidget* cameraWidget = new CameraDisplayWidget(i, this);
        cameraDisplayWidgets.push_back(cameraWidget);
        
        // 添加到网格布局：2行4列
        camerasLayout->addWidget(cameraWidget, row, col);
        
        col++;
        if (col >= 4) {
            col = 0;
            row++;
        }
    }
}
