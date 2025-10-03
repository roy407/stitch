# pragma once
#include <thread>
#include <memory>
#include <string>

class TaskManager {
private:
    std::thread m_thread;
protected:
    std::string m_name;
    bool running{false};
public:
    TaskManager();
    virtual ~TaskManager();
    virtual void start();
    virtual void stop();
    virtual void run() = 0; // 所有子类的run函数都需要优化
    // virtual bool init(void* initMessage); // 延迟初始化，预留接口，待以后修改
private:
};