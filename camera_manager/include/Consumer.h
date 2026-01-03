#pragma once

#include "TaskManager.h"

class Consumer : public TaskManager {
public:
    Consumer();
    virtual ~Consumer();
    virtual void run();
};