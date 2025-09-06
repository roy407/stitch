#include "Consumer.h"

Consumer::Consumer()
{
    m_name += "consumer_";
}

Consumer::~Consumer()
{
}

void Consumer::run()
{
}

bool Consumer::setConsumer(std::weak_ptr<TaskManager> con)
{
    return false;
}
