#pragma once

#include "TaskManager.h"

class Producer : public TaskManager {
public:
    Producer();
    virtual ~Producer();
    virtual void run();
};