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

//TODO : 完全支持 json配置
StitchConsumer *Pipeline::getStitchConsumer(int pipeline_id, std::string kernelTag) {
    LOG_DEBUG("GetPipelineConfig start");
    auto& p = CFG_HANDLE.GetPipelineConfig(pipeline_id);
    LOG_DEBUG("GetPipelineConfig over");
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
        }

    }
    return nullptr;
}

Pipeline::Pipeline(int pipeline_id):Pipeline(CFG_HANDLE.GetPipelineConfig(pipeline_id)){

}

Pipeline::Pipeline(const PipelineConfig &p) {
    LOG_DEBUG("Pipeline:Pipeline start");
    if(p.enable == true) {
        std::vector<FrameChannel*> channels;
        for(auto& cam : p.cameras) {
            std::string type = CFG_HANDLE.GetGlobalConfig().type;
            PacketProducer* pro = nullptr;
            if(type == "mp4") {
                LOG_DEBUG("new MP4PacketProducer satrt");
                pro = new MP4PacketProducer(cam);
                LOG_DEBUG("new MP4PacketProducer over");
            } else if(type == "rtsp") {
                LOG_DEBUG("new RTSPPacketProducer start");
                pro = new RTSPPacketProducer(cam);
                LOG_DEBUG("new RTSPPacketProducer over");
            } else if(type == "usb") {
                LOG_DEBUG("new USBPacketProducer start");
                pro = new USBPacketProducer(cam);
                LOG_DEBUG("new USBPacketProducer over");
            }
            LOG_DEBUG("Pipeline:Pipeline push producer start");
            m_producerTask.push_back(pro);
            LOG_DEBUG("Pipeline:Pipeline push producer over");
            if(m_log) m_log->setProducer(pro);
            if(cam.rtsp == true) {
                RtspConsumer* rtspCon = new RtspConsumer(cam.output_url);
                rtspCon->setChannel(pro->getChannel2Rtsp());
                rtspCon->setParamters(pro->getAVCodecParameters(), pro->getTimeBase());
                m_consumerTask.push_back(rtspCon);
                LOG_DEBUG("Pipeline:Pipeline push rtsp consumer over");
            }
            // TODO: 通过json文件，直接配置好使用的每一个producer或consumer？把初始化过程直接放在配置文件里
            if(CFG_HANDLE.GetGlobalConfig().decoder != "jetson") {
                LOG_DEBUG("new DecoderConsumer");
                DecoderConsumer* dcon = new DecoderConsumer(CFG_HANDLE.GetGlobalConfig().decoder);
                LOG_DEBUG("setAVCodec");
                dcon->setAVCodecParameters(pro->getAVCodecParameters(), pro->getTimeBase());
                LOG_DEBUG("setChannel");
                dcon->setChannel(pro->getChannel2Decoder());
                LOG_DEBUG("Pipeline:Pipeline push decoder consumer satrt");
                m_consumerTask.push_back(dcon);
                LOG_DEBUG("Pipeline:Pipeline push decoder consumer over");
                if(cam.enable_view == true) {
                    SingleViewConsumer* resizeCon = new SingleViewConsumer(cam.width, cam.height, cam.scale_factor);
                    resizeCon->setChannel(dcon->getChannel2Resize());
                    m_resizeStream[cam.cam_id] = resizeCon->getChannel2Show();
                    m_consumerTask.push_back(resizeCon);
                    LOG_DEBUG("Pipeline:Pipeline push resizeCon consumer over");
                }
                channels.push_back(dcon->getChannel2Stitch());
                LOG_DEBUG("Pipeline:Pipeline push getChannel2Stitch over");
            } else {
                JetsonDecoderConsumer* dcon = new JetsonDecoderConsumer();
                dcon->setAVCodecParameters(pro->getAVCodecParameters(), pro->getTimeBase());
                dcon->setChannel(pro->getChannel2Decoder());
                m_consumerTask.push_back(dcon);
                LOG_DEBUG("Pipeline:Pipeline push JetsonDecoder over");
                if(cam.enable_view == true) {
                    SingleViewConsumer* resizeCon = new SingleViewConsumer(cam.width, cam.height, cam.scale_factor);
                    resizeCon->setChannel(dcon->getChannel2Resize());
                    m_resizeStream[cam.cam_id] = resizeCon->getChannel2Show();
                    m_consumerTask.push_back(resizeCon);
                    LOG_DEBUG("Pipeline:Pipeline push resizeCon over");
                }
                channels.push_back(dcon->getChannel2Stitch());
                LOG_DEBUG("Pipeline:Pipeline push getChannel2Stitch over");
            } 
        }
       
        StitchConsumer* stitch = getStitchConsumer(p.pipeline_id, p.stitch.stitch_mode);
        if(stitch != nullptr) {
            stitch->setChannels(channels);
            m_consumerTask.push_back(stitch);
            LOG_DEBUG("Pipeline:Pipeline push stitch over");
            if(m_log) m_log->setConsumer(stitch);

            m_stitchStream = stitch->getChannel2Show();
        } else {
            LOG_INFO("stitch consumer not init");
        }
    }
    LOG_DEBUG("Pipeline:Pipeline over");
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
