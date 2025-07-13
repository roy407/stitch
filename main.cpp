#include <vector>
#include <mutex>
#include <atomic>
#include <iostream>
#include <queue>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <QApplication>

#include "camera_manager.h"
#include "mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
