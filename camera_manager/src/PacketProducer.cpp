#include "RTSPPacketProducer.h"
#include "PacketProducer.h"

void PacketProducer::start() {
    TaskManager::start();
}

void PacketProducer::stop() {
    TaskManager::stop();
}

int PacketProducer::getWidth() const {
    return m_status.width;
}

int PacketProducer::getHeight() const {
    return m_status.height;
}

AVRational PacketProducer::getTimeBase() const {
    return time_base;
}

AVCodecParameters *PacketProducer::getAVCodecParameters() const {
    return codecpar;
}

PacketChannel *PacketProducer::getChannel2Rtsp() const {
    return m_channel2rtsp;
}

PacketChannel *PacketProducer::getChannel2Decoder() const {
    return m_channel2decoder;
}

PacketProducer::PacketProducer() {
    m_channel2decoder = new PacketChannel;
    m_channel2rtsp = new PacketChannel;
    codecpar = avcodec_parameters_alloc();
}

PacketProducer::~PacketProducer() {
    avcodec_parameters_free(&codecpar);
    delete m_channel2rtsp;
    delete m_channel2decoder;
}