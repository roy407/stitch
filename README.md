## 基于GPU的视频拼接项目

## 前置条件

1. nvidia driver >= 535

## 运行步骤如下所示：

1. make stitch_env  #下载所需镜像

2. make build       #程序编译

3. make deploy      #程序打包

## 注：如果只是需要跑代码，可以不进行make deploy这一步，可以make build之后直接cd 进入build目录，运行./stitch_app

# 1、代码结构
      stitch/
      ├── CMakeLists.txt                  # 顶层构建配置
      ├── Makefile                        # 构建与打包入口
      ├── main.cpp                        # 程序主入口
      ├── inc/                            # 全局头文件
      ├── components/                     # UI 模块（Qt 实现）
      │   ├── qt/
      │   └── CMakeLists.txt
      ├── core/                           # 核心逻辑模块
      │   ├── config/                     # 配置读取模块
      │   ├── operator/                   # 异构 算法实现
      │   │   ├── nvidia/
      │   │   ├── ascend/
      │   ├── data_processing/            # 数据处理与拼接流程
      │   ├── utils/                      # 工具类库
      │   │   ├── include/
      │   │   │   ├── log.hpp             # 日志封装
      │   │   │   ├── tools.hpp           # 常用工具
      │   │   │   ├── safe_queue.hpp      # 线程安全队列
      │   │   │   ├── safe_list.h         # 线程安全链表
      │   │   ├── src/
      │   │   └── CMakeLists.txt
      │   └── CMakeLists.txt
      ├── camera_manager/                 # 摄像头与任务管理模块
      ├── scripts/
      │   └── plot_timing.py              # 性能可视化脚本
      ├── resource/
      │   ├── hk5.json
      │   └── hk8.json
      └── README.md
# 2 代码调用过程
## 2.1 main.cpp与lib.cpp
    lib.cpp与main.cpp是独立的，当使用main时,编译生成的stitch_app是可执行文件，而使用lib.cpp时,stitch_app是一个动态链接库（stitch_app.dll）。
    为什么要这样设计？因为之前qt程序是独立的，为了能够显示拼接效果，我们使用qt，通过调用stitch_app.all，创建拼接线程，获取拼接图像。后来将qt和其它程序合并了，直接
  能够获取拼接图像，所以就很少使用lib.cpp了。
    怎么确定使用lib.cpp还是main.cpp?它们的选择方式写在最外层的CMakeLists.txt。也就是：
      if(BUILD_SHARED_LIB)
          add_library(stitch_app SHARED lib.cpp)
      else()
          add_executable(stitch_app main.cpp)
      endif()
## 2.2 从main.cpp到components/qt
    mian.cpp中创建widget对象时调用wiget类的构造函数。进入构造函数后，首先创建了新的Nv12Render对象，它是一个OPENGL渲染器，然后访问单例模式的cam,启动cam，再通
    cam内部的get_stitch_stream（）函数获取图像buffer。再开启一个消费线程(consumerThread())，在消费线程内，先将图像从GPU转换到CPU上，利用行跨度进行32位对齐，将数据拷贝到m_buffer里，使用渲染器进行渲染。
## 2.3 config
    用于读取json文件内的参数内容，是单例模式。
    const GlobalConfig GetGlobalConfig() const;
    const std::vector<CameraConfig> GetCameraConfig() const;
    const GlobalStitchConfig GetGlobalStitchConfig() const;
    分别用于返回json文件中的global，cameras，stitch；
## 2.4 camera_manager
    它是单例模式，用来管理所有生产者消费者的启动，并向外部提供获取拼接图像的接口。
    start()函数中，创建成员格式为TaskManager的vector中，给每个相机创建一个生产者，再创建一个拼接消费者，将他们全部push到vector中，最后再统一启动。外部通过get_stitch_stream()获取到拼接消费者中的图像。
## 2.5 AVFrameProducer
    它继承于Producer类。在构造函数中，创建ffmpeg上下文，设置参数。run函数内打开流获取流信息，并将存储图像和pkt的buffer传入解码器,while循环内打开流获取pkt，一路传给解码器，一路传给RTSPCONSUMER用于推流。
## 2.6 image_decoder
    将AVFrameProducer获取的pkt传给image_decode，通过使用内部函数do_decode解码，再通过m_packetInput存储，外部获取。
## 2.7 StitchConsumer
    每个AVFrameProducer获取的图像buffer一起传给StitchConsumer。在构造中进行了一堆初始化。当它被启动时，在run函数内使用stitch.do_stitch()进行拼接，拼接完后图像被存在frame_output，外部可以通过get_stitch_frame()获取。


