#include <vector>
#include <mutex>
#include <atomic>
#include <iostream>
#include <queue>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <QApplication>
#include "widget.h"
#include "camera_manager.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    Widget w;
    w.show();
    return a.exec();
}
