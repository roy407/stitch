#include <QApplication>
#include "mainwindow.h"
// #include "widget_for_test.h"
#include "camera_manager.h"
#include "config.h"
void launch_with_no_window() {
    camera_manager* cam = camera_manager::GetInstance();
    cam->start();
    cam->stop();
}

// int launch_with_widget(int argc, char *argv[]) {
//     QApplication a(argc, argv);
//     widget_for_test w;
//     w.show();
//     return a.exec();
// }

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
    if(config_name == "resource/cam10") {
        return launch_with_mainwindow(argc, argv);
    } else if(config_name == "resource/hk5") {
        // return launch_with_widget(argc, argv);
    } else {
        launch_with_no_window();
    }
}
