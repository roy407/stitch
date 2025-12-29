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
#include "YUVFrameReader.h"  // 新增头文件

LogConsumer* Pipeline::m_log = nullptr;

//TODO : 完全支持 json配置
StitchConsumer *Pipeline::getStitchConsumer(int pipeline_id, std::string kernelTag) {
    auto& p = CFG_HANDLE.GetPipelineConfig(pipeline_id);
    std::string format =CFG_HANDLE.GetGlobalConfig().format;
    LOG_DEBUG("pipeline id : {}, Format : {}, kernelTag : {}", pipeline_id, format, kernelTag);
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

Pipeline::Pipeline(int pipeline_id):Pipeline(CFG_HANDLE.GetPipelineConfig(pipeline_id)){

}

// 删除或注释掉 #define KERNEL_TEST，使用配置控制
// #define KERNEL_TEST
void Pipeline::SkipDecoder(CameraConfig cam, std::vector<FrameChannel*>& channels)
{
    // ========== 使用YUVFrameReader（跳过解码）==========
    LOG_INFO("Using YUVFrameReader for camera {} (skipping decode)", cam.cam_id);
    
    // 创建YUVFrameReader
    std::string yuv_path;
        
    yuv_path = CFG_HANDLE.GetGlobalConfig().record_path + 
                std::to_string(cam.cam_id) + ".yuv";
    
    LOG_DEBUG("YUV file path for camera {}: {}", cam.cam_id, yuv_path);
    
    std::string format = CFG_HANDLE.GetGlobalConfig().format;

    // 根据format配置设置像素格式
    AVPixelFormat pix_fmt = AV_PIX_FMT_YUV420P;  // 默认
    
    if (format == "YUV420" || format == "NV12") {
        // YUV420通常指NV12格式
        pix_fmt = AV_PIX_FMT_NV12;
        LOG_INFO("Using NV12 format for camera {} (YUV420)", cam.cam_id);
    } else if (format == "YUV420P") {
        pix_fmt = AV_PIX_FMT_YUV420P;
        LOG_INFO("Using YUV420P format for camera {}", cam.cam_id);
    } else if (format == "NV21") {
        pix_fmt = AV_PIX_FMT_NV21;
        LOG_INFO("Using NV21 format for camera {}", cam.cam_id);
    } else {
        LOG_WARN("Unknown format '{}', using default YUV420P", format);
        pix_fmt = AV_PIX_FMT_YUV420P;
    }
    
    YUVFrameReader* yuv_reader = new YUVFrameReader(
        cam.cam_id, 
        yuv_path, 
        cam.width, 
        cam.height,
        pix_fmt
    );
    


    // 添加到任务列表
    m_producerTask.push_back(yuv_reader);
    
    // 设置日志
    if(m_log) {
        // LogConsumer适配代码
    }
    
    // RTSP输出（YUV数据需要编码后才能输出RTSP）
    if(cam.rtsp == true) {
        LOG_WARN("RTSP output requires encoding. YUV input cannot directly output RTSP for camera {}", 
                cam.cam_id);
    }
    
    // 单视图显示
    if(cam.enable_view == true) {
        LOG_DEBUG("Enabling single view for camera {}", cam.cam_id);
        SingleViewConsumer* resizeCon = new SingleViewConsumer(cam.width, cam.height, cam.scale_factor);
        resizeCon->setChannel(yuv_reader->getChannel2Resize());
        m_resizeStream[cam.cam_id] = resizeCon->getChannel2Show();
        m_consumerTask.push_back(resizeCon);
    }
    
    // 将拼接通道添加到channels列表
    channels.push_back(yuv_reader->getChannel2Stitch());
}

void Pipeline::NormalDecoder(CameraConfig cam, std::vector<FrameChannel*>& channels)
{
    // ========== 原始的解码流程 ==========
    LOG_INFO("Using original decode pipeline for camera {}", cam.cam_id);
    
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
        LOG_DEBUG("Enabling RTSP output for camera {}", cam.cam_id);
        RtspConsumer* rtspCon = new RtspConsumer(cam.output_url);
        rtspCon->setChannel(pro->getChannel2Rtsp());
        rtspCon->setParamters(pro->getAVCodecParameters(), pro->getTimeBase());
        m_consumerTask.push_back(rtspCon);
    }
    
    // 选择解码器
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
        channels.push_back(dcon->getChannel2Stitch());
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
        channels.push_back(dcon->getChannel2Stitch());
    }
}
Pipeline::Pipeline(const PipelineConfig &p) {
    if(p.enable == true) {
        std::vector<FrameChannel*> channels;
        
        // 获取全局配置中的输入类型
        std::string input_type = CFG_HANDLE.GetGlobalConfig().type;
        std::string format = CFG_HANDLE.GetGlobalConfig().format;
        
        LOG_INFO("Pipeline using input_type: {}, format: {}", input_type, format);
        
        for(auto& cam : p.cameras) {
            LOG_DEBUG("Processing camera {} with input type: {}, format: {}", 
                     cam.cam_id, input_type, format);
            
            // 根据输入类型选择不同的数据源
            if (input_type == "yuv" || input_type == "YUV") 
            {
                SkipDecoder(cam,channels);
            } 
            else 
            {
                NormalDecoder(cam,channels);
            }
        }
       
        // 创建拼接消费者
        StitchConsumer* stitch = getStitchConsumer(p.pipeline_id, p.stitch.stitch_mode);
        if(stitch != nullptr) {
            stitch->setChannels(channels);
            m_consumerTask.push_back(stitch);
            if(m_log) m_log->setConsumer(stitch);
            m_stitchStream = stitch->getChannel2Show();
        } else {
            LOG_ERROR("Failed to initialize stitch consumer");
        }
    } else {
        LOG_INFO("Pipeline {} is disabled", p.pipeline_id);
    }
}

void Pipeline::setLogConsumer(LogConsumer *log) {
    m_log = log;
}

void Pipeline::start() {
    LOG_INFO("Starting pipeline...");
    for(auto& pro : m_producerTask) pro->start();
    for(auto& con : m_consumerTask) con->start();
}

void Pipeline::stop() {
    LOG_INFO("Stopping pipeline...");
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