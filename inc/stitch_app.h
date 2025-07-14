#pragma once

#include "safe_queue.hpp"
#include <libavutil/frame.h>  // for AVFrame


safe_queue<std::pair<AVFrame*,costTimes>>& launch_stitch_worker();

