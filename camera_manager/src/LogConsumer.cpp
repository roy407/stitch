#include "LogConsumer.h"
#include <cuda_runtime.h>
#include <chrono>
#include "log.hpp"
#include "AVFrameProducer.h"
#include "StitchConsumer.h"
#include <unistd.h>

struct CpuStats {
    unsigned long long user, nice, system, idle, iowait, irq, softirq, steal;
};

void LogConsumer::printProducer(AVFrameProducer *pro, uint64_t& prev_frame_cnt) {
    if(pro) {
        CamStatus tmp = pro->m_status;
        LOG_INFO("{} [{},{}] FPS:{}", 
            pro->m_name, 
            tmp.width, 
            tmp.height, 
            (tmp.frame_cnt - prev_frame_cnt) / 2);
        prev_frame_cnt = tmp.frame_cnt;
    }
}

void LogConsumer::printConsumer(StitchConsumer *con, uint64_t& prev_frame_cnt) {
    if(con) {
        StitchStatus tmp = con->m_status;
        LOG_INFO("{} [{},{}] FPS:{}", 
            con->m_name, 
            tmp.width, 
            tmp.height, 
            (tmp.frame_cnt - prev_frame_cnt) / 2);
        prev_frame_cnt = tmp.frame_cnt;
    }
}

void LogConsumer::printGPUStatus() {
    int dev;
    cudaGetDevice(&dev);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);

    LOG_INFO("GPU: {}, Mem {}MB/{}MB used", 
        prop.name, 
        (totalMem - freeMem) / (1024 * 1024), 
        totalMem / (1024 * 1024));
}

void LogConsumer::printCPUStatus() {
    auto read_cpu_stats = []() -> CpuStats {
        std::ifstream file("/proc/stat");
        std::string cpu;
        CpuStats stats{};
        file >> cpu >> stats.user >> stats.nice >> stats.system >> stats.idle
            >> stats.iowait >> stats.irq >> stats.softirq >> stats.steal;
        return stats;
    };
    auto get_cpu_usage = [&]() -> double {
        CpuStats t1 = read_cpu_stats();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CpuStats t2 = read_cpu_stats();

        unsigned long long idle1 = t1.idle + t1.iowait;
        unsigned long long idle2 = t2.idle + t2.iowait;

        unsigned long long non_idle1 = t1.user + t1.nice + t1.system + t1.irq + t1.softirq + t1.steal;
        unsigned long long non_idle2 = t2.user + t2.nice + t2.system + t2.irq + t2.softirq + t2.steal;

        unsigned long long total1 = idle1 + non_idle1;
        unsigned long long total2 = idle2 + non_idle2;

        double totald = (double)(total2 - total1);
        double idled = (double)(idle2 - idle1);

        return (totald - idled) / totald * 100.0;
    };

    auto get_mem_usage = []() -> double {
        std::ifstream file("/proc/meminfo");
        std::string key;
        unsigned long long value;
        std::string unit;
        unsigned long long mem_total = 0, mem_free = 0, buffers = 0, cached = 0;

        while (file >> key >> value >> unit) {
            if (key == "MemTotal:") mem_total = value;
            else if (key == "MemFree:") mem_free = value;
            else if (key == "Buffers:") buffers = value;
            else if (key == "Cached:") cached = value;
        }
        unsigned long long used = mem_total - mem_free - buffers - cached;
        return (double)used / mem_total * 100.0;
    };
    double cpu = get_cpu_usage();
    double mem = get_mem_usage();
    LOG_INFO("CPU Usage: {}%, Memory Usage: {}%", cpu, mem);
}

LogConsumer::LogConsumer() {
}

LogConsumer::~LogConsumer() {

}

void LogConsumer::start() {
    TaskManager::start();
}

void LogConsumer::stop() {
    TaskManager::stop();
}

void LogConsumer::run() {
    // 上限为20个摄像头
    uint64_t prev_frame_cnt[32] = {};
    while(running) {
        std::this_thread::sleep_for(std::chrono::seconds(time_gap));
        m_time += 2;
        LOG_INFO("============ Frame Statistics ============");
        for(int i=0;i<m_pro.size();i++) printProducer(m_pro[i], prev_frame_cnt[i]);
        printConsumer(m_con, prev_frame_cnt[31]);
        printGPUStatus();
        printCPUStatus();
    }
}

void LogConsumer::setProducer(AVFrameProducer *pro) {
    m_pro.push_back(pro);
}

void LogConsumer::setConsumer(StitchConsumer *con) {
    m_con = con;
}
