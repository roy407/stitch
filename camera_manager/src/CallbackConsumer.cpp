#include "CallbackConsumer.h"

void CallbackConsumer::setChannel(FrameChannel *channel) {
    m_channel = channel;
}

void CallbackConsumer::setCallback(void (*callback)(Frame)) {
    m_callback = callback;
}

CallbackConsumer::CallbackConsumer() {
    m_name += "callback";
}

CallbackConsumer::~CallbackConsumer() {

}

void CallbackConsumer::start() {
    TaskManager::start();
}

void CallbackConsumer::stop() {
    m_channel->stop();
    TaskManager::stop();
}

void CallbackConsumer::run() {
    while(running) {
        Frame frame;
        if(!m_channel->recv(frame)) break;
        m_callback(frame);
    }
    m_channel->clear();
}
