主程序进去
直接调用widget显示窗口
调用进入qt的事件循环
Widget::Widget(QWidget *parent) : 
    QOpenGLWidget(parent),开始初始化成员变量，设置qt日志过滤
    之后
    之后创建了NV12render渲染器，获得camera——manager的单例，首次调用创建静态单例对象，获得配置实例，读取相机配置数量，config进行单例从初始化
    加载json的配置文件
    启动camera_manager接下来初始化网络模块，创建LogConsumer
    创建日志消费者，用于统计和打印性能信息
    之后循环创建AVFrameProducer为每个相机都创建新的生产者，之后获取每个相机的高和宽，将producer 关联log，将produce加入任务的列表，保留produce的帧序列
    之后到生产者里面，对每一路摄像头设置相机ID 设置名称，之后到分配ffmpeg的格式上下文，到设置rtsp传输选项，设置超时，防止崩溃，之后到设置读取状态。从配置中读取款核稿
    之后开始创建stitchConsumer，创建拼接消费者，接收所有的producer的帧队列，保存帧对象列表
    设置名称，获取相机数量
    保存输入宽和高
    计算输出宽度
    初始化拼接器
    从配置之中去读输出URL
    分配rtsp输出上下文
    创建输出流
    设置流参数
    拼接器初始化
保存参数，计算出输出宽度，读取裁剪配置并计算出裁剪区域数组，分配cuda内存：单应性矩阵，Y平面指针数组， uv平面指针数组，裁剪区域，之后便是行跨度数组。创建cuda硬件帧的上下文，初始化硬件帧的上下文。设置运行标志。
cuda——handle——init单例出事haul
创建静态单例，触发构造函数cuda_handle_init构造函数，创建cuda硬件设备上下文。创建cuda硬件的上下文，之后camera_manager::start() - 启动所有任务依次启动所有TaskManager（LogConsumer, AVFrameProducer们, StitchConsumer）
回到widget构造函数，获取拼接流，widget构造函数，创建消费线程。 Widget构造函数结束
下面开始OpenGL初始化
设置着色器: `setupShaders()`
设置几何体: `setupGeometry()`
设置纹理: `setupTextures()`
6生产者线程运行阶段
AVFrameProducer线程启动，在独立的线程之中，打开输入流，查找流的信息，获取流和解码器参数，启动图像解码器，创建rtsp的消费者，关闭输入流。
将编解码器参数赋值到上下文，打开编解码器，保存输出队列，设置运行标志，启动解码线程。
解码线程中 从队列之中等待数据包，发送数据包到解码器，循环接收解码后的帧，分配帧，接收帧，把cuda格式帧推送到队列
释放数据包。
打开输入流
循环读取帧，克隆数据包，推送到rtsp队列，推送到解码队列，更新帧计数，释放数据包，关闭输入流
拼接消费者线程运行阶段
stitchconsumer线程启动
分配输入帧数组
之后的话，从每个producer队列等待一帧
执行拼接设置时间戳。推送到输出队列。
更新帧计数。释放输入帧。
执行拼接
定义单应性矩阵数组，提取GPU内存指针，分配输出帧，设置输出帧格式，引用硬件帧上下文，获取GPU缓冲区，拷贝单应性矩阵到GPU，拷贝输入指针数组到GPU 提取输出指针，提取输入行跨度。拷贝行跨度到gpu
启动cuda拼接内核，同步cuda流，返回输出帧
rtsp消费者线程
从队列等待数据包，写入rtsp流，释放数据包
9程序关闭阶段
设置停止标志
停止队列，等待消费线程，删除消费线程，停止camera——manager 删除渲染器，停止所有任务，清理网络模块，设置停止标志，等待线程结束。记录日志。停止rtsp消费者，停止taskmanager 关闭图像解码器。设置停止标志，停止队列，等待解码线程。释放硬件设备上下文，释放编解码上下文。
1. **m_packetSender1**: AVFrameProducer → RtspConsumer (数据包)
2. **m_packetSender2**: AVFrameProducer → image_decoder (数据包)
3. **m_frameSender**: image_decoder → StitchConsumer (GPU帧)
4. **frame_output**: StitchConsumer → Widget消费线程 (拼接后的GPU帧)