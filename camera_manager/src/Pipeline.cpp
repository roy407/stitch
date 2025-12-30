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
Pipeline::Pipeline(const PipelineConfig &p) {
    if(p.enable == true) {
        std::vector<FrameChannel*> channels;
        
        // 获取全局配置中的输入类型
        std::string input_type = CFG_HANDLE.GetGlobalConfig().type;  // 新增配置项
        
        for(auto& cam : p.cameras) {
            LOG_DEBUG("Processing camera {} with input type: {}", cam.cam_id, input_type);
            
            // 根据输入类型选择不同的数据源
            if (input_type == "yuv" || input_type == "YUV") {
                // ========== 使用YUVFrameReader（跳过解码）==========
                LOG_INFO("Using YUVFrameReader for camera {} (skipping decode)", cam.cam_id);
                
                // 创建YUVFrameReader
                std::string yuv_path;
                    
                yuv_path = CFG_HANDLE.GetGlobalConfig().record_path + 
                            std::to_string(cam.cam_id) + ".yuv";
                
                
                LOG_DEBUG("YUV file path for camera {}: {}", cam.cam_id, yuv_path);
                
                YUVFrameReader* yuv_reader = new YUVFrameReader(
                    cam.cam_id, 
                    yuv_path, 
                    cam.width, 
                    cam.height
                );
                
                // 设置帧率（从配置读取或使用默认值）
                if (cam.test_fps > 0) {
                    // 这里需要在YUVFrameReader中添加setFPS方法
                    // yuv_reader->setFPS(cam.test_fps);
                }
                
                // 添加到任务列表
                m_producerTask.push_back(yuv_reader);
                
                // 设置日志（如果需要）
                if(m_log) {
                    // 注意：LogConsumer可能需要适配YUVFrameReader
                    // m_log->setProducer(yuv_reader);
                }
                
                // RTSP输出（如果需要，但YUV数据需要先编码）
                if(cam.rtsp == true) {
                    LOG_WARN("RTSP output not supported for YUV input (needs encoding)");
                    // 如果需要RTSP输出，需要添加编码器
                }
                
                // 单视图显示
                if(cam.enable_view == true) {
                    LOG_DEBUG("xxxxxxxxxxxview truexxxxxxxxxx");
                    SingleViewConsumer* resizeCon = new SingleViewConsumer(cam.width, cam.height, cam.scale_factor);
                    resizeCon->setChannel(yuv_reader->getChannel2Resize());
                    m_resizeStream[cam.cam_id] = resizeCon->getChannel2Show();
                    m_consumerTask.push_back(resizeCon);
                }
                
                // 将拼接通道添加到channels列表
                channels.push_back(yuv_reader->getChannel2Stitch());
                
            } else {
                // ========== 原始的解码流程 ==========
                LOG_INFO("Using original decode pipeline for camera {}", cam.cam_id);
                
                std::string type = CFG_HANDLE.GetGlobalConfig().type;
                PacketProducer* pro = nullptr;
                
                #ifndef KERNEL_TEST
                if(type == "mp4") {
                    pro = new MP4PacketProducer(cam);
                } else if(type == "rtsp") {
                    pro = new RTSPPacketProducer(cam);
                } else if(type == "usb") {
                    pro = new USBPacketProducer(cam);
                }
                #else
                // 使用 TestPacketProducer
                pro = new TestPacketProducer(cam);
                TestPacketProducer* testProducer = dynamic_cast<TestPacketProducer*>(pro);
                if (testProducer) {
                    if (cam.test_fps > 0) {
                        testProducer->setFrameRate(cam.test_fps);
                    }
                    if (cam.test_pattern >= 0) {
                        testProducer->setTestPattern(cam.test_pattern);
                    }
                }
                #endif
                
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