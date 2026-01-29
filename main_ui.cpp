#include <QApplication>
#include "mainwindow.h"
#include "widget_for_test.h"
#include "config.h"
#include <iostream>

int launch_with_mainwindow(int argc, char *argv[]) {
    QApplication a(argc, argv);
    StitchMainWindow w;
    w.show();
    return a.exec();
}

int launch_with_widget(int pipeline_id, int width, int height, int argc, char *argv[]) {
    QApplication a(argc, argv);
    widget_for_test w(pipeline_id, width, height);
    w.show();
    return a.exec();
}

int main(int argc, char *argv[]) {
    std::string config_name = "";
    if (argc > 1) {
        config_name = argv[1];
    }

    // Load config to determine layout
    config::SetConfigFileName(config_name);

    if (config_name.find("cam10.json") != std::string::npos) {
        return launch_with_mainwindow(argc, argv);
    } else if(config_name.find(".json") != std::string::npos) {
         // for cam5, cam2 etc, use the simpler widget which we updated to support SHM
         // Assuming default resolution for test widget
         return launch_with_widget(0, 1920, 540, argc, argv);
    } else {
         std::cout << "Usage: ./stitch_ui <config.json>" << std::endl;
         std::cout << "Example: ./stitch_ui resource/cam10.json" << std::endl;
         return -1;
    }
}
