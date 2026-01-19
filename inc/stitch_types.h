#pragma once

#include <stdint.h>
#include <cstddef> 

#ifdef __cplusplus
extern "C" {
#endif

const int TYPES_MAX_CAM_SIZE=10;

/* ================== costTimes ================== */

// 使用不透明指针模式
typedef struct types_costTimes types_costTimes_t;

/* ================== Frame ================== */

typedef struct types_Frame types_Frame_t;

/* ================== Frame ================== */

// typedef struct types_Frame {
//     int cam_id;
//     types_AVFrame_t* m_data;
//     types_costTimes_t cost_times;
//     uint64_t timestamp;
// } types_Frame_t;

#ifdef __cplusplus
}
#endif
