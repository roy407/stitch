#include "LogConsumer.h"
#include <cuda_runtime.h>
#include <chrono>
#include "log.hpp"
#include "RTSPPacketProducer.h"
#include "StitchConsumer.h"
#include <unistd.h>
#include <dirent.h> 

struct CpuStats {
    unsigned long long user, nice, system, idle, iowait, irq, softirq, steal;
};

float LogConsumer::CalculateProFPS(PacketProducer* pro, uint64_t& prev_frame_cnt, uint64_t& prev_timestamp)
{
    CamStatus tmp = pro->m_status;
    float fps=(tmp.frame_cnt - prev_frame_cnt) / ((tmp.timestamp - prev_timestamp) * 1e-9);
    prev_frame_cnt = tmp.frame_cnt;
    prev_timestamp = tmp.timestamp;
    return fps;
}

void LogConsumer::printProducer(PacketProducer *pro, float fps) {
    if(pro) {
        CamStatus tmp = pro->m_status;
        LOG_INFO("{} [{},{}] FPS:{:.2f}", 
            pro->m_name, 
            tmp.width, 
            tmp.height, 
            fps);
    }
}

void LogConsumer::printConsumer(StitchConsumer *con, uint64_t& prev_frame_cnt, uint64_t& prev_timestamp) {
    if(con) {
        StitchStatus tmp = con->m_status;
        LOG_INFO("{} [{},{}] FPS:{:.2f}", 
            con->m_name, 
            tmp.width, 
            tmp.height, 
            (tmp.frame_cnt - prev_frame_cnt) / ((tmp.timestamp - prev_timestamp) * 1e-9));
        prev_frame_cnt = tmp.frame_cnt;
        prev_timestamp = tmp.timestamp;
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
    LOG_INFO("CPU Usage: {:.2f}%, Memory Usage: {:.2f}%", cpu, mem);
}


void LogConsumer::monitorCPU_Core() {
    // 读取两个时间点的CPU核心数据
    auto read_all_core_stats = []() -> std::vector<CpuStats> {
        std::vector<CpuStats> cores;
        std::ifstream stat_file("/proc/stat");
        std::string line;
        
        while (std::getline(stat_file, line)) {
            if (line.compare(0, 3, "cpu") == 0) {
                // cpu0, cpu1, cpu2... 但第一个是总的cpu统计
                static int core_index = -1;
                core_index++;
                
                if (core_index == 0) continue;  // 跳过总的cpu统计
                
                std::istringstream iss(line);
                std::string cpu_label;
                CpuStats stats{};
                
                iss >> cpu_label >> stats.user >> stats.nice >> stats.system 
                    >> stats.idle >> stats.iowait >> stats.irq >> stats.softirq 
                    >> stats.steal;
                
                cores.push_back(stats);
            }
        }
        return cores;
    };
    
    // 读取第一个时间点的数据
    std::vector<CpuStats> t1 = read_all_core_stats();
    
    // 等待一小段时间
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // 读取第二个时间点的数据
    std::vector<CpuStats> t2 = read_all_core_stats();
    
    // 计算每个核心的使用率
    std::string core_usage_str;
    for (size_t i = 0; i < std::min(t1.size(), t2.size()); ++i) {
        const CpuStats& c1 = t1[i];
        const CpuStats& c2 = t2[i];
        
        // 计算空闲时间
        unsigned long long idle1 = c1.idle + c1.iowait;
        unsigned long long idle2 = c2.idle + c2.iowait;
        
        // 计算非空闲时间
        unsigned long long non_idle1 = c1.user + c1.nice + c1.system + 
                                       c1.irq + c1.softirq + c1.steal;
        unsigned long long non_idle2 = c2.user + c2.nice + c2.system + 
                                       c2.irq + c2.softirq + c2.steal;
        
        // 计算总时间
        unsigned long long total1 = idle1 + non_idle1;
        unsigned long long total2 = idle2 + non_idle2;
        
        double totald = static_cast<double>(total2 - total1);
        double idled = static_cast<double>(idle2 - idle1);
        
        double usage = 0.0;
        if (totald > 0) {
            usage = (totald - idled) / totald * 100.0;
        }
        
        core_usage_str += fmt::format("Core{} : {:.1f}%  ;", i, usage);
    }
    
    if (!core_usage_str.empty()) {
        LOG_INFO("CPU Core Usage: {}", core_usage_str);
    }
}

void LogConsumer::monitorMainThreads() {
    // 获取所有线程ID
    std::vector<int> thread_ids;
    
    // 打开线程目录
    const char* task_dir_path = "/proc/self/task";
    DIR* task_dir = opendir(task_dir_path);
    
    if (!task_dir) {
        LOG_ERROR("Failed to open task directory: {}", task_dir_path);
        return;
    }
    
    // 遍历目录项
    struct dirent* entry;
    while ((entry = readdir(task_dir)) != nullptr) {
        // 跳过 . 和 ..
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        
        // 转换为线程ID
        int tid = atoi(entry->d_name);
        if (tid > 0) {
            thread_ids.push_back(tid);
        }
    }
    
    closedir(task_dir);
    
    // 为每个线程计算CPU使用率（需要两次采样）
    static std::map<int, std::pair<long, long>> prev_cpu_times;
    
    // 第一次采样
    std::map<int, std::pair<long, long>> current_cpu_times;
    for (int tid : thread_ids) {
        std::string stat_path = "/proc/self/task/" + std::to_string(tid) + "/stat";
        std::ifstream stat_file(stat_path);
        if (stat_file) {
            std::string line;
            std::getline(stat_file, line);
            std::istringstream iss(line);
            
            std::string ignore;
            long utime, stime;
            iss >> ignore >> ignore >> ignore;  // pid, name, state
            for (int i = 0; i < 11; ++i) iss >> ignore;
            iss >> utime >> stime;
            
            current_cpu_times[tid] = {utime, stime};
        }
    }
    
    // 等待一小段时间
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // 获取时钟滴答频率（静态变量，只需获取一次）
    static long clock_ticks_per_sec = sysconf(_SC_CLK_TCK);
    
    // 获取当前时间（用于计算实际经过的时间）
    static auto last_calc_time = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    double elapsed_seconds = std::chrono::duration<double>(now - last_calc_time).count();
    last_calc_time = now;
    
    // 记录有活动的线程
    std::vector<std::tuple<int, std::string, double>> active_threads;
    
    for (int tid : thread_ids) {
        std::string stat_path = "/proc/self/task/" + std::to_string(tid) + "/stat";
        std::ifstream stat_file(stat_path);
        if (stat_file) {
            std::string line;
            std::getline(stat_file, line);
            std::istringstream iss(line);
            
            std::string ignore, name;
            long utime2, stime2;
            iss >> ignore >> name >> ignore;  // pid, name, state
            for (int i = 0; i < 11; ++i) iss >> ignore;
            iss >> utime2 >> stime2;
            
            // 计算CPU使用率
            double cpu_usage = 0;
            if (prev_cpu_times.find(tid) != prev_cpu_times.end()) {
                auto [utime1, stime1] = prev_cpu_times[tid];
                long total_diff = (utime2 + stime2) - (utime1 + stime1);
                
                // 正确计算公式：
                // CPU使用率 = (CPU时间差(秒) / 时间间隔(秒)) × 100%
                // CPU时间差(秒) = total_diff / clock_ticks_per_sec
                if (elapsed_seconds > 0 && clock_ticks_per_sec > 0) {
                    double cpu_seconds = static_cast<double>(total_diff) / clock_ticks_per_sec;
                    cpu_usage = (cpu_seconds / elapsed_seconds) * 100.0;
                    
                    // 注意：这里计算的是相对于单个CPU核心的使用率
                    // 如果要考虑多核，可以乘以核心数
                    static int num_cores = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
                    cpu_usage *= num_cores;
                }
            }
            
            // 清理线程名
            if (name.length() > 2 && name.front() == '(' && name.back() == ')') {
                name = name.substr(1, name.length() - 2);
            }
            
            // 只记录CPU使用率大于0.1%的线程
            if (cpu_usage > 0.1) {
                active_threads.emplace_back(tid, name, cpu_usage);
            }
        }
    }
    
    // 打印有活动的线程
    if (!active_threads.empty()) {
        LOG_INFO("=== Thread CPU Usage (averaged over {:.3f}s) ===", elapsed_seconds);
        LOG_DEBUG("Total threads: {}", thread_ids.size());
        // 按CPU使用率排序
        std::sort(active_threads.begin(), active_threads.end(),
                  [](const auto& a, const auto& b) {
                      return std::get<2>(a) > std::get<2>(b);
                  });
        
        for (const auto& [tid, name, cpu_usage] : active_threads) {
            LOG_INFO("Thread {} [{}]: CPU {:.2f}%", tid, name, cpu_usage);
        }
    }
    
    // 更新上一次的CPU时间
    prev_cpu_times = current_cpu_times;
}

void LogConsumer::detectThreadBlocks() {
    struct ThreadState {
        std::string name;
        char state;
        unsigned long long cpu_time;
        std::chrono::steady_clock::time_point last_update;
        int d_state_counter;  // D状态计数器
        int low_cpu_counter;  // 低CPU计数器
    };
    
    static std::map<int, ThreadState> thread_states;
    static auto last_check_time = std::chrono::steady_clock::now();
    
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - last_check_time).count();
    
    if (elapsed < 1.0) return;  // 每秒检查一次
    
    // 获取当前线程状态
    std::map<int, ThreadState> current_states;
    
    DIR* dir = opendir("/proc/self/task");
    if (!dir) return;
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] == '.') continue;
        
        int tid = atoi(entry->d_name);
        if (tid <= 0) continue;
        
        std::string stat_path = "/proc/self/task/" + std::to_string(tid) + "/stat";
        std::ifstream stat_file(stat_path);
        if (!stat_file) continue;
        
        std::string line;
        std::getline(stat_file, line);
        std::istringstream iss(line);
        
        std::string pid_str, name;
        char state;
        unsigned long long utime = 0, stime = 0;
        
        iss >> pid_str >> name >> state;
        for (int i = 0; i < 11; ++i) { std::string dummy; iss >> dummy; }
        iss >> utime >> stime;
        
        if (name.length() > 2 && name[0] == '(' && name.back() == ')') {
            name = name.substr(1, name.length() - 2);
        }
        
        ThreadState ts;
        ts.name = name;
        ts.state = state;
        ts.cpu_time = utime + stime;
        ts.last_update = now;
        
        // 从历史中获取计数器
        auto it = thread_states.find(tid);
        if (it != thread_states.end()) {
            const ThreadState& old = it->second;
            
            // 检查D状态
            if (state == 'D') {
                ts.d_state_counter = old.d_state_counter + 1;
            } else {
                ts.d_state_counter = 0;
            }
            
            // 检查CPU使用率是否变化
            if (ts.cpu_time == old.cpu_time) {
                ts.low_cpu_counter = old.low_cpu_counter + 1;
            } else {
                ts.low_cpu_counter = 0;
            }
            
            // 检查是否可能阻塞
            if (state == 'D' && ts.d_state_counter >= 3) {
                LOG_WARN("Thread {} [{}] in D state for {} checks - possible I/O blockage", 
                        tid, name, ts.d_state_counter);
            } else if (ts.low_cpu_counter >= 5 && state != 'R') {
                LOG_WARN("Thread {} [{}] no CPU activity for {} checks - possible blockage", 
                        tid, name, ts.low_cpu_counter);
            }
        } else {
            ts.d_state_counter = 0;
            ts.low_cpu_counter = 0;
        }
        
        current_states[tid] = ts;
    }
    closedir(dir);
    
    thread_states = std::move(current_states);
    last_check_time = now;
}

//LogConsumer继承于Consumer,Consumer又继承于TaskManager，初始化从最内层往外
LogConsumer::LogConsumer() {
    LOG_DEBUG("LogConsumer start");
    m_name += "log";
    LOG_DEBUG("LogConsumer over");
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
    uint64_t prev_timestamp[32] = {};
    static int first_broken=0;
    while(running) {
        std::this_thread::sleep_for(std::chrono::seconds(time_gap));
        m_time += 2;
        float fps;
        int normal=1;
        LOG_INFO("============ Frame Statistics ============");
        for(int i=0;i<m_pro.size();i++)
        {
            fps=CalculateProFPS(m_pro[i],prev_frame_cnt[i],prev_timestamp[i]);
            printProducer(m_pro[i], fps);
            if(fps>10)
                ;
            else
                normal=0;
        }
        if(normal)
        {
            for(int i=0;i<m_con.size();i++) printConsumer(m_con[i], prev_frame_cnt[21 + i], prev_timestamp[21 + i]);
            printGPUStatus();
            printCPUStatus();
            // monitorCPU_Core();
            monitorMainThreads();
            detectThreadBlocks();
        }
        else
        {
            if(first_broken>0)
                LOG_WARN("Link broken.Trying to reconnect......");
            else
                first_broken=1;
        }
    }
}

void LogConsumer::setProducer(PacketProducer *pro) {
    m_pro.push_back(pro);
}

void LogConsumer::setConsumer(StitchConsumer *con) {
    m_con.push_back(con);
}
