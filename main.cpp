#include <QApplication>
#include "mainwindow.h"
#include "widget_for_test.h"
#include "camera_manager.h"
#include "config.h"
void launch_with_no_window() {
    camera_manager* cam = camera_manager::GetInstance();
    cam->start();
    while(1);
}

int launch_with_widget(int pipeline_id, int width, int height, int argc, char *argv[]) {
    QApplication a(argc, argv);
    widget_for_test w(pipeline_id, width, height);
    w.show();
    return a.exec();
}

int launch_with_mainwindow(int argc, char *argv[]) {
    QApplication a(argc, argv);
    StitchMainWindow w;
    w.show();
    return a.exec();
}

int main(int argc, char *argv[]) {
    std::string config_name = "";
    if (argc > 1) {
        config_name = argv[1];
    }
    config::SetConfigFileName(config_name);
    if(config_name == "resource/cam10.json") {
        return launch_with_mainwindow(argc, argv);
    } else if(config_name == "resource/hk5.json") {
        return launch_with_widget(0, 1920, 540, argc, argv);
    } else if(config_name == "resource/cam5.json") {
        return launch_with_widget(0, 1920, 540, argc, argv);
    } else if(config_name == "resource/cam2.json") {
        return launch_with_widget(0, 1920, 540, argc, argv);
    } else if(config_name == "resource/cam10_jetson.json") {
        return launch_with_widget(0, 1920, 540, argc, argv);
    } else {
        launch_with_no_window();
    }
}
