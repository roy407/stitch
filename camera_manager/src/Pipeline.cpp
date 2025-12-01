#include "Pipeline.h"
#include "RtspConsumer.h"
#include "ResizeConsumer.h"
#include "DecoderConsumer.h"
#include "StitchImpl.h"
#include "StitchConsumer.h"

#define cameras_debug

#if defined(cameras_debug)
#include "AVFrameProducer_debug.h"
#endif

LogConsumer* Pipeline::m_log = nullptr;

//TODO : 完全支持 json配置
StitchOps *Pipeline::getStitchOps(int pipeline_id, std::string Format, std::string kernelTag) {
    auto& p = CFG_HANDLE.GetPipelineConfig(pipeline_id);
    if(Format == "YUV420") {
        if(kernelTag == "mapping_table") {
        auto stitchImpl = new StitchImpl<YUV420, MappingTableKernel>();
        stitchImpl->loadMappingTable(p.stitch.stitch_impl.mapping_table.d_mapping_table);
        StitchOps* ops = make_stitch_ops(stitchImpl);
        ops->init(ops->obj, p.cameras.size(), p.cameras[0].width, p.stitch.stitch_impl.mapping_table.output_width, p.default_height);
        return ops;
        } else {

        }
    } else {

    }
    return nullptr;
}

// TODO: 构造函数待补充
Pipeline::Pipeline(int pipeline_id) {

}

Pipeline::Pipeline(const PipelineConfig &p) {
    if(p.enable == true) {
        std::vector<FrameChannel*> channels;
        for(auto& cam : p.cameras) {
            #if defined(cameras_debug)
            AVFrameProducer_debug* pro = new AVFrameProducer_debug(cam);
            #else
            AVFrameProducer* pro = new AVFrameProducer(cam);
            #endif
            m_producerTask.push_back(pro);
            if(m_log) m_log->setProducer(pro);
            if(cam.rtsp == true) {
                RtspConsumer* rtspCon = new RtspConsumer(cam.output_url);
                rtspCon->setChannel(pro->getChannel2Rtsp());
                rtspCon->setParamters(pro->getAVCodecParameters(), pro->getTimeBase());
                m_consumerTask.push_back(rtspCon);
            }
            DecoderConsumer* dcon = new DecoderConsumer(CFG_HANDLE.GetGlobalConfig().decoder);
            dcon->setAVCodecParameters(pro->getAVCodecParameters(), pro->getTimeBase());
            dcon->setChannel(pro->getChannel2Decoder());
            m_consumerTask.push_back(dcon);
            if(cam.resize == true) {
                ResizeConsumer* resizeCon = new ResizeConsumer(cam.width, cam.height, cam.scale_factor);
                resizeCon->setChannel(dcon->getChannel2Resize());
                m_resizeStream[cam.cam_id] = resizeCon->getChannel2Show();
                m_consumerTask.push_back(resizeCon);
            }
            channels.push_back(dcon->getChannel2Stitch());
        }
        // TODO: YUV420 放置在json中
        StitchConsumer* stitch = new StitchConsumer(getStitchOps(p.pipeline_id, "YUV420", p.stitch.stitch_mode),
            p.cameras[0].width, p.default_height, p.stitch.stitch_impl.mapping_table.output_width);
        stitch->setChannels(channels);
        m_consumerTask.push_back(stitch);
        if(m_log) m_log->setConsumer(stitch);

        m_stitchStream = stitch->getChannel2Show();
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
        LOG_WARN("can't find resize camera stream, cam_id is {}", cam_id);
        return nullptr;
    }
}

size_t Pipeline::getResizeCameraStreamCount() const {
    return m_resizeStream.size();
}
