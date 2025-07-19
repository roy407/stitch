#include <vector>
#include <mutex>
#include <atomic>
#include <iostream>
#include <queue>
#include <chrono>
#include <acl/acl.h>

#include "camera_manager.h"

int main() {
    aclError ret = aclInit(NULL);
    if (ret != ACL_SUCCESS) {
        const char* err_msg = aclGetRecentErrMsg();
        std::cerr << "ACL init failed: " << err_msg << std::endl;
    }

    camera_manager* camera = camera_manager::GetInstance();
    camera->start();
    camera->join();

    ret = aclFinalize();
    if (ret != ACL_SUCCESS) {
        const char* err_msg = aclGetRecentErrMsg();
        std::cerr << "ACL finalize failed: " << err_msg << std::endl;
    }

    std::cout<<__func__<<" exit!"<<std::endl;
    return 0;
}
