#include <QApplication>
#include "widget.h"
#include "mainwindow.h"
#include "camera_manager.h"

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
    MainWindow w;
    w.show();
    return a.exec();
}

int main(int argc, char *argv[]) {
    return launch_with_widget(argc, argv);
}
