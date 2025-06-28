#pragma once

#include "safe_queue.hpp"
#include <libavutil/frame.h>  // for AVFrame


safe_queue<AVFrame*>& launch_stitch_worker();

