#include "Pipeline.h"
#include "RtspConsumer.h"
#include "SingleViewConsumer.h"
#include "JetsonDecoderConsumer.h"
#include "DecoderConsumer.h"
#include "StitchImpl.h"
#include "StitchConsumer.h"
#include "RTSPPacketProducer.h"
#include "MP4PacketProducer.h"
#include "USBPacketProducer.h"

LogConsumer* Pipeline::m_log = nullptr;

/**
 * @brief 根据 pipeline_id 和 kernelTag 创建并返回对应的 StitchConsumer
 * 
 * 这个函数是一个工厂函数，用于创建不同格式和算法的拼接消费者。它根据配置中的
 * 图像格式（YUV420/YUV420P）和指定的 kernel 类型，实例化对应的 StitchImpl
 * 模板类，并将其包装成 StitchConsumer 返回。
 * 
 * 函数逻辑：
 * 1. 根据 pipeline_id 获取对应的配置信息
 * 2. 根据全局配置中的图像格式和 kernelTag 选择不同的实现
 * 3. 每种组合创建对应的 StitchImpl 模板实例
 * 4. 初始化 StitchOps 并创建 StitchConsumer
 * 
 * 支持的格式：
 *   - YUV420：YUV420格式（NV12等）
 *   - YUV420P：YUV420平面格式（三个分离的平面）
 * 
 * 支持的 kernel 类型：
 *   - mapping_table：使用映射表进行拼接
 *   - raw：原始拼接（无变换）
 *   - h_matrix_inv：使用单应性矩阵逆变换进行拼接
 *   - h_matrix_inv_v1_1：单应性矩阵逆变换v1.1版本
 *   - h_matrix_inv_v2：单应性矩阵逆变换v2版本
 * 
 * 使用示例：
 * @code
 * // 创建使用映射表的 YUV420 拼接消费者
 * StitchConsumer* consumer1 = pipeline->getStitchConsumer(0, "mapping_table");
 * 
 * // 创建使用 H 矩阵逆变换的 YUV420P 拼接消费者
 * StitchConsumer* consumer2 = pipeline->getStitchConsumer(1, "h_matrix_inv");
 * 
 * @endcode
 * 
 * 注意：
 *   - 函数返回的 StitchConsumer 需要由调用者管理生命周期
 *   - 如果没有匹配的格式和 kernel 组合，函数返回 nullptr
 *   - 确保传入的 pipeline_id 有效，否则配置获取可能失败
 * 
 * @param pipeline_id 管道ID，用于获取对应的配置
 * @param kernelTag 内核标签，指定要使用的拼接算法类型
 * @return StitchConsumer* 返回创建的拼接消费者指针，如果创建失败返回 nullptr
 * 
 * @see StitchImpl, StitchOps, StitchConsumer
 */

StitchConsumer *Pipeline::getStitchConsumer(int pipeline_id, std::string kernelTag) {
    auto& p = CFG_HANDLE.GetPipelineConfig(pipeline_id);
    std::string format =CFG_HANDLE.GetGlobalConfig().format;
    LOG_INFO("pipeline id : {}, Format : {}, kernelTag : {}", pipeline_id, format, kernelTag);
    if(format == "YUV420") {        
        if(kernelTag =="mapping_table" ) {
            auto stitchImpl = new StitchImpl<YUV420, MappingTableKernel>();
            stitchImpl->loadMappingTable(p.stitch.stitch_impl.mapping_table.d_mapping_table);
            StitchOps* ops = make_stitch_ops(stitchImpl);
            ops->init(ops->obj, p.cameras.size(), p.cameras[0].width, p.stitch.stitch_impl.mapping_table.output_width, p.default_height);
            return new StitchConsumer(ops, p.cameras[0].width, p.default_height, p.stitch.stitch_impl.mapping_table.output_width);
        } else if(kernelTag == "raw") {
            auto stitchImpl = new StitchImpl<YUV420, RawKernel>();
            StitchOps* ops = make_stitch_ops(stitchImpl);
            ops->init(ops->obj, p.cameras.size(), p.cameras[0].width, p.default_width, p.default_height);
            return new StitchConsumer(ops, p.cameras[0].width, p.default_height, p.default_width);;
        } else if (kernelTag == "h_matrix_inv") {
            auto stitchImpl = new StitchImpl<YUV420, HMatrixInvKernel>();
            stitchImpl->loadConfig(p.stitch.stitch_impl, p.cameras.size());
            StitchOps* ops = make_stitch_ops(stitchImpl);
            ops->init(ops->obj, p.cameras.size(), p.cameras[0].width, p.default_width, p.default_height);
            return new StitchConsumer(ops, p.cameras[0].width, p.default_height, p.default_width);

        } else if (kernelTag == "h_matrix_inv_v1_1") {
            auto stitchImpl = new StitchImpl<YUV420, HMatrixInvV1_1Kernel>();
            stitchImpl->loadConfig(p.stitch.stitch_impl, p.cameras.size());
            StitchOps* ops = make_stitch_ops(stitchImpl);
            ops->init(ops->obj, p.cameras.size(), p.cameras[0].width, p.default_width, p.default_height);
            return new StitchConsumer(ops, p.cameras[0].width, p.default_height, p.default_width);
           
        } else if (kernelTag == "h_matrix_inv_v2") {
            auto stitchImpl = new StitchImpl<YUV420, HMatrixInvV2Kernel>();
            stitchImpl->loadConfig(p.stitch.stitch_impl, p.cameras.size());
            StitchOps* ops = make_stitch_ops(stitchImpl);
            ops->init(ops->obj, p.cameras.size(), p.cameras[0].width, p.default_width, p.default_height);
            return new StitchConsumer(ops, p.cameras[0].width, p.default_height, p.default_width);
            
        }
    } else if(format == "YUV420P") {
        if(kernelTag =="mapping_table" ) {
            auto stitchImpl = new StitchImpl<YUV420P, MappingTableKernel>();
            stitchImpl->loadMappingTable(p.stitch.stitch_impl.mapping_table.d_mapping_table);
            StitchOps* ops = make_stitch_ops(stitchImpl);
            ops->init(ops->obj, p.cameras.size(), p.cameras[0].width, p.stitch.stitch_impl.mapping_table.output_width, p.default_height);
            return new StitchConsumer(ops, p.cameras[0].width, p.default_height, p.stitch.stitch_impl.mapping_table.output_width);
        } else if(kernelTag == "raw") {
            auto stitchImpl = new StitchImpl<YUV420P, RawKernel>();
            StitchOps* ops = make_stitch_ops(stitchImpl);
            ops->init(ops->obj, p.cameras.size(), p.cameras[0].width, p.default_width, p.default_height);
            return new StitchConsumer(ops, p.cameras[0].width, p.default_height, p.default_width);;
        } else if (kernelTag == "h_matrix_inv") {
            auto stitchImpl = new StitchImpl<YUV420P, HMatrixInvKernel>();
            stitchImpl->loadConfig(p.stitch.stitch_impl, p.cameras.size());
            StitchOps* ops = make_stitch_ops(stitchImpl);
            ops->init(ops->obj, p.cameras.size(), p.cameras[0].width, p.default_width, p.default_height);
            return new StitchConsumer(ops, p.cameras[0].width, p.default_height, p.default_width);
            
        } else if (kernelTag == "h_matrix_inv_v1_1") {
            auto stitchImpl = new StitchImpl<YUV420P, HMatrixInvV1_1Kernel>();
            stitchImpl->loadConfig(p.stitch.stitch_impl, p.cameras.size());
            StitchOps* ops = make_stitch_ops(stitchImpl);
            ops->init(ops->obj, p.cameras.size(), p.cameras[0].width, p.default_width, p.default_height);
            return new StitchConsumer(ops, p.cameras[0].width, p.default_height, p.default_width);
           
        } else if (kernelTag == "h_matrix_inv_v2") {
            auto stitchImpl = new StitchImpl<YUV420P, HMatrixInvV2Kernel>();
            stitchImpl->loadConfig(p.stitch.stitch_impl, p.cameras.size());
            StitchOps* ops = make_stitch_ops(stitchImpl);
            ops->init(ops->obj, p.cameras.size(), p.cameras[0].width, p.default_width, p.default_height);
            return new StitchConsumer(ops, p.cameras[0].width, p.default_height, p.default_width);

        }
        

    }
    return nullptr;
}


/**
 * @brief 初始化相机数据处理流
 *
 * 为每个相机创建数据处理管道，包括生产者、解码器、视图显示等组件，
 * 并根据配置连接它们。函数返回每个相机用于后续拼接的通道。
 *
 * @param p 包含相机配置和全局设置的管道配置
 * @return FrameChannel* 每个相机用于拼接的通道
 */
FrameChannel* Pipeline::initCameraProcessingFlows(const CameraConfig &cam) {
    std::string type = CFG_HANDLE.GetGlobalConfig().type;
    PacketProducer* pro = nullptr;
    if(type == "mp4") {
        pro = new MP4PacketProducer(cam);
    } else if(type == "rtsp") {
        pro = new RTSPPacketProducer(cam);
    } else if(type == "usb") {
        pro = new USBPacketProducer(cam);
    }
    m_producerTask.push_back(pro);
    if(m_log) m_log->setProducer(pro);
    if(cam.rtsp == true) {
        RtspConsumer* rtspCon = new RtspConsumer(cam.output_url);
        rtspCon->setChannel(pro->getChannel2Rtsp());
        rtspCon->setParamters(pro->getAVCodecParameters(), pro->getTimeBase());
        m_consumerTask.push_back(rtspCon);
    }
    // TODO: 通过json文件，直接配置好使用的每一个producer或consumer？把初始化过程直接放在配置文件里
    if(CFG_HANDLE.GetGlobalConfig().decoder != "jetson") {
        DecoderConsumer* dcon = new DecoderConsumer(CFG_HANDLE.GetGlobalConfig().decoder);
        dcon->setAVCodecParameters(pro->getAVCodecParameters(), pro->getTimeBase());
        dcon->setChannel(pro->getChannel2Decoder());
        m_consumerTask.push_back(dcon);
        if(cam.enable_view == true) {
            SingleViewConsumer* resizeCon = new SingleViewConsumer(cam.width, cam.height, cam.scale_factor);
            resizeCon->setChannel(dcon->getChannel2Resize());
            m_resizeStream[cam.cam_id] = resizeCon->getChannel2Show();
            m_consumerTask.push_back(resizeCon);
        }
        return dcon->getChannel2Stitch();
    } else {
        JetsonDecoderConsumer* dcon = new JetsonDecoderConsumer();
        dcon->setAVCodecParameters(pro->getAVCodecParameters(), pro->getTimeBase());
        dcon->setChannel(pro->getChannel2Decoder());
        m_consumerTask.push_back(dcon);
        if(cam.enable_view == true) {
            SingleViewConsumer* resizeCon = new SingleViewConsumer(cam.width, cam.height, cam.scale_factor);
            resizeCon->setChannel(dcon->getChannel2Resize());
            m_resizeStream[cam.cam_id] = resizeCon->getChannel2Show();
            m_consumerTask.push_back(resizeCon);
        }
        return dcon->getChannel2Stitch();
    }
    
}

Pipeline::Pipeline(int pipeline_id):Pipeline(CFG_HANDLE.GetPipelineConfig(pipeline_id)){

}

Pipeline::Pipeline(const PipelineConfig &p) {
    if(p.enable == true) {
        std::vector<FrameChannel*> channels;
        #if !defined(KERNEL_TEST)
        for(auto& cam : p.cameras) {
            channels.push_back(initCameraProcessingFlows(cam));
        }
        #else
        std::string format =CFG_HANDLE.GetGlobalConfig().format;
        for(auto& cam : p.cameras) {
            Frame f;
            f.cam_id = cam.cam_id;
            f.m_data = get_frame_on_gpu_memory(format, cam.width, cam.height, cuda_handle_init::GetGPUDeviceHandle());
            FrameChannel* fc = new FrameChannel;
            fc->send(f);
            channels.push_back(fc);
        }
        #endif
        StitchConsumer* stitch = getStitchConsumer(p.pipeline_id, p.stitch.stitch_mode);
        if(stitch != nullptr) {
            stitch->setChannels(channels);
            m_consumerTask.push_back(stitch);
            if(m_log) m_log->setConsumer(stitch);

            m_stitchStream = stitch->getChannel2Show();
        } else {
            LOG_INFO("stitch consumer not init");
        }
    }
}

void Pipeline::setLogConsumer(LogConsumer *log) {
    m_log = log;
}

void Pipeline::start() {
    for(auto& pro : m_producerTask) pro->start();
    for(auto& con : m_consumerTask) con->start();
}

void Pipeline::stop() {
    m_resizeStream.clear();
    for(auto& con : m_consumerTask) con->stop();
    for(auto& pro : m_producerTask) pro->stop();
}

FrameChannel* Pipeline::getStitchCameraStream() const {
    return m_stitchStream;
}

FrameChannel *Pipeline::getResizeCameraStream(int cam_id) const {
    if(m_resizeStream.find(cam_id) != m_resizeStream.end()) {
        return m_resizeStream.at(cam_id);
    } else {
        LOG_WARN("can't find enable_view camera stream, cam_id is {}", cam_id);
        return nullptr;
    }
}

size_t Pipeline::getResizeCameraStreamCount() const {
    return m_resizeStream.size();
}
