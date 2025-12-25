#include "LogConsumer.h"
#include <cuda_runtime.h>
#include <chrono>
#include "log.hpp"
#include "RTSPPacketProducer.h"
#include "StitchConsumer.h"
#include <unistd.h>
#include <dirent.h> 
#include <nvml.h>

struct CpuStats {
    unsigned long long user, nice, system, idle, iowait, irq, softirq, steal;
};

void LogConsumer::printNvidiaEncoderDecoderStatus() {
    static bool nvml_initialized = false;
    static nvmlReturn_t last_result;
    
    // 初始化NVML
    if (!nvml_initialized) {
        last_result = nvmlInit();
        if (NVML_SUCCESS != last_result) {
            LOG_ERROR("Failed to initialize NVML: {}", nvmlErrorString(last_result));
            return;
        }
        nvml_initialized = true;
    }
    
    // 获取GPU句柄
    nvmlDevice_t device;
    last_result = nvmlDeviceGetHandleByIndex(0, &device);
    if (NVML_SUCCESS != last_result) {
        LOG_ERROR("Failed to get device handle: {}", nvmlErrorString(last_result));
        return;
    }
    
    // 1. 获取编码器使用率（可用）
    unsigned int encoder_utilization = 0;
    unsigned int encoder_sampling_period_us = 0;
    
    last_result = nvmlDeviceGetEncoderUtilization(device, &encoder_utilization, 
                                                 &encoder_sampling_period_us);
    if (NVML_SUCCESS == last_result) {
        LOG_INFO("NVIDIA Encoder: {}% (sampling: {}us)", 
                encoder_utilization, encoder_sampling_period_us);
    } else {
        LOG_DEBUG("Encoder utilization failed: {}", nvmlErrorString(last_result));
    }
    
    // 2. 获取解码器使用率（可用）
    unsigned int decoder_utilization = 0;
    unsigned int decoder_sampling_period_us = 0;
    
    last_result = nvmlDeviceGetDecoderUtilization(device, &decoder_utilization, 
                                                 &decoder_sampling_period_us);
    if (NVML_SUCCESS == last_result) {
        LOG_INFO("NVIDIA Decoder: {}% (sampling: {}us)", 
                decoder_utilization, decoder_sampling_period_us);
    } else {
        LOG_DEBUG("Decoder utilization failed: {}", nvmlErrorString(last_result));
    }
    
    // 3. 获取编码器统计信息（可用）
    unsigned int encoder_session_count = 0;
    unsigned int encoder_avg_fps = 0;
    unsigned int encoder_avg_latency = 0;
    
    last_result = nvmlDeviceGetEncoderStats(device, &encoder_session_count, 
                                           &encoder_avg_fps, &encoder_avg_latency);
    if (NVML_SUCCESS == last_result) {
        LOG_INFO("Encoder Stats: Sessions={}, Avg FPS={}, Avg Latency={}ms", 
                encoder_session_count, encoder_avg_fps, encoder_avg_latency);
    }
    
    // 4. 获取编码器容量（可用）
    nvmlEncoderType_t encoder_query_type = NVML_ENCODER_QUERY_H264;
    unsigned int encoder_capacity = 0;
    
    last_result = nvmlDeviceGetEncoderCapacity(device, encoder_query_type, &encoder_capacity);
    if (NVML_SUCCESS == last_result) {
        LOG_INFO("Encoder Available Capacity: {}%", encoder_capacity);
    }
    
    // 注意：以下函数在你的驱动版本中不可用，已跳过
    // - nvmlDeviceGetDecoderStats
    // - nvmlDeviceGetEncoderSessionCount  
    // - nvmlDeviceGetDecoderSessionCount
}

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

void LogConsumer::monitorThreads() {
    // 静态变量保存线程历史
    struct ThreadHistory {
        std::string name;
        uint64_t last_total_ticks;
        std::chrono::steady_clock::time_point last_time;
        double last_cpu_percent;
        char last_state;                     // 上一次的状态
        int d_state_counter;                 // D状态持续次数
        int inactive_counter;                // 不活跃计数器
        int same_state_counter;              // 相同状态持续次数
        std::chrono::steady_clock::time_point last_state_change; // 上次状态变化时间
        bool reported_blocked;               // 是否已报告阻塞
    };
    
    static std::map<int, ThreadHistory> thread_histories;
    static auto last_print_time = std::chrono::steady_clock::now();
    static std::vector<std::string> blocked_threads_summary; // 阻塞线程摘要
    
    auto now = std::chrono::steady_clock::now();
    auto elapsed_since_print = std::chrono::duration_cast<std::chrono::seconds>(now - last_print_time).count();
    
    // 每2秒打印一次
    if (elapsed_since_print < 2) return;
    
    last_print_time = now;
    
    // 获取系统时钟滴答频率
    static long clock_ticks_per_second = sysconf(_SC_CLK_TCK);
    if (clock_ticks_per_second <= 0) clock_ticks_per_second = 100;
    
    // 收集当前所有线程
    std::vector<std::tuple<int, std::string, double, char, bool>> current_threads; // 添加是否阻塞标志
    std::vector<std::string> current_blocked_threads; // 当前检测到的阻塞线程
    
    DIR* task_dir = opendir("/proc/self/task");
    if (!task_dir) {
        LOG_WARN("Failed to open /proc/self/task");
        return;
    }
    
    struct dirent* entry;
    while ((entry = readdir(task_dir)) != nullptr) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        
        int tid = atoi(entry->d_name);
        if (tid <= 0) continue;
        
        // 读取线程状态文件
        std::string stat_path = "/proc/self/task/" + std::to_string(tid) + "/stat";
        std::ifstream stat_file(stat_path);
        if (!stat_file) {
            continue;
        }
        
        std::string line;
        std::getline(stat_file, line);
        stat_file.close();
        
        // 使用空格分割整个行
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;
        
        while (iss >> token) {
            tokens.push_back(token);
        }
        
        if (tokens.size() < 15) {
            LOG_DEBUG("Stat line too short for TID {}: {} tokens", tid, tokens.size());
            continue;
        }
        
        // 提取线程名
        std::string comm = tokens[1];
        if (comm.length() > 2 && comm.front() == '(' && comm.back() == ')') {
            comm = comm.substr(1, comm.length() - 2);
        }
        
        // 提取状态
        char state = tokens[2].empty() ? '?' : tokens[2][0];
        
        // 提取utime和stime
        uint64_t utime = 0, stime = 0;
        try {
            utime = std::stoull(tokens[13]);
            stime = std::stoull(tokens[14]);
        } catch (const std::exception& e) {
            LOG_DEBUG("Failed to parse CPU times for TID {}: {}", tid, e.what());
            continue;
        }
        
        uint64_t total_cpu_ticks = utime + stime;
        
        // 计算CPU使用率
        double cpu_percent = 0.0;
        bool is_blocked = false;
        std::string block_reason;
        
        auto it = thread_histories.find(tid);
        if (it != thread_histories.end()) {
            ThreadHistory& history = it->second;
            
            // 计算时间差（秒）
            double time_diff = std::chrono::duration<double>(now - history.last_time).count();
            
            if (time_diff > 0) {
                // 计算CPU时间差（时钟滴答）
                uint64_t cpu_ticks_diff = 0;
                if (total_cpu_ticks >= history.last_total_ticks) {
                    cpu_ticks_diff = total_cpu_ticks - history.last_total_ticks;
                } else {
                    // 处理可能的计数器回绕
                    cpu_ticks_diff = (std::numeric_limits<uint64_t>::max() - history.last_total_ticks) + total_cpu_ticks;
                }
                
                // 转换为秒
                double cpu_seconds = static_cast<double>(cpu_ticks_diff) / clock_ticks_per_second;
                
                // CPU使用率百分比（相对于单个核心）
                cpu_percent = (cpu_seconds / time_diff) * 100.0;
                
                // 平滑处理
                cpu_percent = 0.7 * cpu_percent + 0.3 * history.last_cpu_percent;
                
                // 检测CPU不活跃
                if (cpu_percent < 0.1) {
                    history.inactive_counter++;
                } else {
                    history.inactive_counter = 0;
                }
            }
            
            // 检测状态变化
            if (state != history.last_state) {
                history.last_state = state;
                history.same_state_counter = 1;
                history.last_state_change = now;
                history.d_state_counter = (state == 'D' || state == 'U') ? 1 : 0;
            } else {
                history.same_state_counter++;
                if (state == 'D' || state == 'U') {
                    history.d_state_counter++;
                }
            }
            
            // 阻塞检测逻辑
            // 1. D/U状态（不可中断睡眠）持续3次检查（约6秒）
            if (history.d_state_counter >= 3) {
                is_blocked = true;
                block_reason = fmt::format("D/U state for {} checks (I/O blockage)", history.d_state_counter);
            }
            // 2. 长时间不活跃（CPU使用率<0.1%）且不是睡眠状态
            else if (history.inactive_counter >= 5 && state != 'S' && state != 'D') {
                is_blocked = true;
                block_reason = fmt::format("No CPU activity for {} checks", history.inactive_counter);
            }
            // 3. 长时间处于相同状态（非运行状态）且CPU使用率低
            else if (history.same_state_counter >= 10 && state != 'R' && cpu_percent < 0.5) {
                auto state_duration = std::chrono::duration_cast<std::chrono::seconds>(
                    now - history.last_state_change).count();
                if (state_duration > 10) {
                    is_blocked = true;
                    block_reason = fmt::format("Stuck in state '{}' for {} seconds", state, state_duration);
                }
            }
            // 4. 运行状态但CPU使用率极低（可能自旋等待或饥饿）
            else if (state == 'R' && cpu_percent < 0.1 && history.inactive_counter >= 3) {
                is_blocked = true;
                block_reason = fmt::format("Running but no CPU usage for {} checks (possible spin wait)", history.inactive_counter);
            }
            
            // 如果检测到阻塞且未报告过
            if (is_blocked && !history.reported_blocked) {
                LOG_WARN("Thread {} [{}] may be blocked: {}", tid, comm, block_reason);
                history.reported_blocked = true;
                current_blocked_threads.push_back(fmt::format("{} [{}]: {}", tid, comm, block_reason));
            } else if (!is_blocked && history.reported_blocked) {
                LOG_INFO("Thread {} [{}] is no longer blocked", tid, comm);
                history.reported_blocked = false;
            }
            
            // 更新历史记录
            history.last_total_ticks = total_cpu_ticks;
            history.last_time = now;
            history.last_cpu_percent = cpu_percent;
        } else {
            // 第一次看到这个线程，初始化历史记录
            ThreadHistory history;
            history.name = comm;
            history.last_total_ticks = total_cpu_ticks;
            history.last_time = now;
            history.last_cpu_percent = 0.0;
            history.last_state = state;
            history.d_state_counter = (state == 'D' || state == 'U') ? 1 : 0;
            history.inactive_counter = 0;
            history.same_state_counter = 1;
            history.last_state_change = now;
            history.reported_blocked = false;
            thread_histories[tid] = history;
            
            // 第一次不计算CPU使用率
            cpu_percent = 0.0;
        }
        
        current_threads.emplace_back(tid, comm, cpu_percent, state, is_blocked);
    }
    closedir(task_dir);
    
    // 清理已经不存在的线程
    auto it = thread_histories.begin();
    while (it != thread_histories.end()) {
        bool found = false;
        for (const auto& [tid, name, cpu, state, blocked] : current_threads) {
            if (tid == it->first) {
                found = true;
                break;
            }
        }
        if (!found) {
            if (it->second.reported_blocked) {
                LOG_INFO("Blocked thread {} [{}] has terminated", it->first, it->second.name);
            }
            it = thread_histories.erase(it);
        } else {
            ++it;
        }
    }
    
    // 更新阻塞线程摘要
    if (!current_blocked_threads.empty()) {
        blocked_threads_summary = current_blocked_threads;
    }
    
    // 打印结果
    LOG_INFO("========== Thread CPU Usage ===========");
    LOG_INFO("Threads: {} total", current_threads.size());
    
    // 按CPU使用率降序排序
    std::sort(current_threads.begin(), current_threads.end(),
             [](const auto& a, const auto& b) {
                 return std::get<2>(a) > std::get<2>(b);
             });
    
    LOG_INFO("%CPU   TID    STATE   NAME (B=Blocked)");
    // R (Running): 线程正在运行或可运行（在运行队列中）。
    // S (Sleeping): 线程正在可中断的睡眠中（等待某个事件完成，如I/O操作、信号等）。
    // D (Disk Sleep): 线程正在不可中断的睡眠中（通常是在等待I/O，如磁盘I/O）。在这种状态下，线程不会响应信号。
    // T (Stopped): 线程被停止（通常是由于收到SIGSTOP、SIGTSTP、SIGTTIN、SIGTTOU信号）。
    // t (Tracing stop): 线程被调试器暂停（例如，通过ptrace）。
    // Z (Zombie): 僵尸线程，已终止但尚未被父线程回收。
    // X (Dead): 线程已死亡（几乎从不会看到，因为死亡状态非常短暂）。
    // I (Idle): 空闲线程（在较新的内核中，空闲线程的状态为I，但注意这个状态并不常见于普通线程）。

    int count = 0;
    static int show_count = 10;
    double total_cpu_all_threads = 0.0;
    int blocked_count = 0;
    
    for (const auto& [tid, name, cpu_percent, state, blocked] : current_threads) {
        if (count++ >= show_count) break;
        
        total_cpu_all_threads += cpu_percent;
        if (blocked) blocked_count++;
        
        // 格式化输出，阻塞线程用特殊标记
        std::string blocked_marker = blocked ? " [B]" : "";
        LOG_INFO("{:5.1f}%   {:6}   {:1}      {}{}", 
                cpu_percent, tid, state, name, blocked_marker);
    }
    
    if (current_threads.size() > show_count) {
        LOG_INFO("... and {} more threads", current_threads.size() - show_count);
    }
    
    //这个线程的总CPU使用率计算方法和硬件CPU的占用率计算方法不一样，这个甚至可以超过100%
    LOG_INFO("Total CPU usage: {:.1f}% across all threads", total_cpu_all_threads);
    
    // 打印阻塞线程摘要
    static int block_check_counter = 0;
    if (++block_check_counter % 5 == 0) { // 每10秒（5次检查）打印一次阻塞摘要
        if (!blocked_threads_summary.empty()) {
            LOG_WARN("======= Thread Block Detection Summary =======");
            LOG_WARN("Found {} potentially blocked threads:", blocked_threads_summary.size());
            for (const auto& blocked_info : blocked_threads_summary) {
                LOG_WARN("  {}", blocked_info);
            }
            
        } else if (blocked_count > 0) {
            LOG_INFO("Currently {} threads marked as blocked", blocked_count);
        } else {
            LOG_INFO("No blocked threads detected");
        }
    }
    
    // 额外统计信息
    static int stat_counter = 0;
    if (++stat_counter % 10 == 0) { // 每20秒打印一次详细统计
        LOG_DEBUG("=== Thread Monitoring Statistics ===");
        LOG_DEBUG("Tracking {} threads in history", thread_histories.size());
        
        // 统计各种状态的线程数
        std::map<char, int> state_counts;
        for (const auto& [tid, history] : thread_histories) {
            state_counts[history.last_state]++;
        }
        
        LOG_DEBUG("Thread state distribution:");
        for (const auto& [state, count] : state_counts) {
            LOG_DEBUG("  State '{}': {} threads", state, count);
        }
    }
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
        for(int i=0;i<m_con.size();i++) printConsumer(m_con[i], prev_frame_cnt[21 + i], prev_timestamp[21 + i]);
        LOG_INFO("=========Hardware Usage=========");
        printGPUStatus();
        printNvidiaEncoderDecoderStatus();
        printCPUStatus();
        // monitorCPU_Core();
        monitorThreads();
        if(normal)
        {
            ;
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
