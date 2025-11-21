#include <QApplication>
#include "mainwindow.h"
// #include "widget.h"
#include "camera_manager.h"
#include "config.h"
void launch_with_no_window() {
    camera_manager* cam = camera_manager::GetInstance();
    cam->start();
    while(1);
}

int launch_with_widget(int argc, char *argv[]) {
    QApplication a(argc, argv);
    Widget w;
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
    return launch_with_mainwindow(argc, argv);
}
