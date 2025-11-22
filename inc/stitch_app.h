#pragma once

#include "safe_queue.hpp"
#include <libavutil/frame.h>  // for AVFrame

safe_queue<T_Frame>& launch_stitch_worker();

