#include <vector>
#include <mutex>
#include <atomic>
#include <iostream>
#include <queue>
#include <chrono>
<<<<<<< Updated upstream
#include <cuda.h>
#include <cuda_runtime.h>
#include <QApplication>
=======
#include <acl/acl.h>
>>>>>>> Stashed changes

#include "camera_manager.h"
#include "mainwindow.h"

<<<<<<< Updated upstream
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
=======
int main() {
    aclError ret = aclInit(NULL);
    if (ret != ACL_SUCCESS) {
        const char* err_msg = aclGetRecentErrMsg();
        std::cerr << "ACL init failed: " << err_msg << std::endl;
    }

    camera_manager camera;
    camera.start();

    ret = aclFinalize();
    if (ret != ACL_SUCCESS) {
        const char* err_msg = aclGetRecentErrMsg();
        std::cerr << "ACL finalize failed: " << err_msg << std::endl;
    }

    std::cout<<__func__<<" exit!"<<std::endl;
    return 0;
}
>>>>>>> Stashed changes
