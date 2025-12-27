# LogConsumer编写说明

1.
LogConsumer::monitorThreads()是线程监视函数，其实是将以下命令写成程序
ps -ef | grep stitch
top -H -p <PID>     #<PID>是当前进程的PID值

程序实现是读取Linux系统的"/proc/self/task"文件，
此文件里保存的有各个线程的状态信息，文件中各个信息是用"."或".."等隔开的，需要将其格式化，提取各个变量。
而CPU占用率的计算方法是，每间隔一段时间读取所有进程的文件，而每次线程启动的时候都会在线程文件里更新其时间信息，用线程的启动时间间隔除以采样时间，就是CPU占用率。
每个线程的CPU使用率是相对于单个CPU核心的，如果系统有多个CPU核心，所有线程的总CPU使用率可能超过100%。
线程记录要及时清理，防止无效线程的不断增长，
而且Linux 线程ID (TID) 可能被重用，如果不清理：
时刻1：ThreadA (TID=1001) 存在，被监控
时刻2：ThreadA 终止，但历史记录保留
时刻3：ThreadB (TID=1001) 被创建（相同TID）
时刻4：监控函数会误用 ThreadA 的历史数据来计算 ThreadB 的CPU使用率
这会导致错误的CPU百分比计算和状态判断
如果阻塞线程恢复了，他的状态也会被更新，恢复监测

2.
void LogConsumer::printNvidiaEncoderDecoderStatus()
这个是检测英伟达硬件编解码的状态的，而受限于英伟达驱动的版本，有些检测函数是无法使用的，
所以用下面的程序先检测了哪些检测函数可以使用，如果有不能用的函数要重新检测。
int main() {
    // 动态加载NVML库
    void* handle = dlopen("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1", RTLD_LAZY);
    if (!handle) {
        printf("Failed to load NVML library: %s\n", dlerror());
        return 1;
    }
    
    // 获取NVML版本
    typedef nvmlReturn_t (*nvmlSystemGetNVMLVersion_t)(unsigned int*, unsigned int);
    nvmlSystemGetNVMLVersion_t getVersion = 
        (nvmlSystemGetNVMLVersion_t)dlsym(handle, "nvmlSystemGetNVMLVersion");
    
    if (getVersion) {
        unsigned int version;
        nvmlReturn_t result = getVersion(&version, sizeof(version));
        if (result == NVML_SUCCESS) {
            printf("NVML Version: %u.%u.%u\n", 
                   version / 1000, (version % 1000) / 10, version % 10);
        }
    }
    
    // 检查各种函数是否存在
    #define CHECK_FUNC(func) \
        printf("%-40s: %s\n", #func, dlsym(handle, #func) ? "AVAILABLE" : "NOT AVAILABLE")
    
    printf("\n=== Function Availability ===\n");
    CHECK_FUNC(nvmlDeviceGetEncoderUtilization);
    CHECK_FUNC(nvmlDeviceGetDecoderUtilization);
    CHECK_FUNC(nvmlDeviceGetEncoderStats);
    CHECK_FUNC(nvmlDeviceGetDecoderStats);
    CHECK_FUNC(nvmlDeviceGetEncoderCapacity);
    CHECK_FUNC(nvmlDeviceGetEncoderSessionCount);
    CHECK_FUNC(nvmlDeviceGetDecoderSessionCount);
    
    dlclose(handle);
    return 0;
}

3.
void LogConsumer::monitorCPU_Core()监测各个CPU核心的使用情况，信息有些冗余，没有启用，
可以在LogConsumer::run()里启用