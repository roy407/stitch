#pragma once

#include <stdint.h>
#include <cstddef> 

#ifdef __cplusplus
extern "C" {
#endif

const int MAX_CAM_SIZE=10;

/* ================== costTimes ================== */

typedef struct costTimes {
    uint64_t image_frame_cnt[MAX_CAM_SIZE];
    uint64_t when_get_packet[MAX_CAM_SIZE];
    uint64_t when_get_decoded_frame[MAX_CAM_SIZE];
    uint64_t when_get_stitched_frame;
    uint64_t when_show_on_the_screen;
} stitch_cost_times_t;

struct AVFrame;

/* ================== Frame ================== */

typedef struct Frame {
    int cam_id;
    AVFrame* m_data;
    stitch_cost_times_t cost_times;
    uint64_t timestamp;
} stitch_frame_t;

#ifdef __cplusplus
}
#endif
