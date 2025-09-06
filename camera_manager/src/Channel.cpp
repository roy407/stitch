#include "Channel.h"

bool FrameChannel::bind(TaskManager *producer, TaskManager *consumer, ChannelType type)
{
    if(producer == nullptr || consumer == nullptr) return false;
    Channel* pro = producer->GetChannel();
    Channel* con = consumer->GetChannel();
    if(type == P2C) {
        con->m_data = pro->m_data;
    } else if(type == C2P) {
        pro->m_data = con->m_data;
    } else {
        return false;
    }
}

bool PacketChannel::bind(TaskManager *producer, TaskManager *consumer, ChannelType type)
{
    if(producer == nullptr || consumer == nullptr) return false;
    Channel* pro = producer->GetChannel();
    Channel* con = consumer->GetChannel();
    if(type == P2C) {
        con->m_data = pro->m_data;
    } else if(type == C2P) {
        pro->m_data = con->m_data;
    } else {
        return false;
    }
}