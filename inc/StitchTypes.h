#pragma once

#include <stdint.h>
#include <cstddef> 

#ifdef __cplusplus
extern "C" {
#endif

struct AVFrame;
struct AVPacket;

#define MAX_CAM_SIZE  10

typedef enum {
    STITCH_OK            = 0,   // 成功
    STITCH_ERR_INVALID   = -1,  // 参数非法 / handle 无效 / 未初始化
    STITCH_ERR_CONFIG    = -2,  // 配置文件相关错误
    STITCH_ERR_RELOAD    = -3   // 重新加载配置失败
} STITCH_STATUS;

typedef struct costTimes {
    uint64_t image_frame_cnt[MAX_CAM_SIZE] = {};
    uint64_t when_get_packet[MAX_CAM_SIZE] = {};
    uint64_t when_get_decoded_frame[MAX_CAM_SIZE] = {};
    uint64_t when_get_stitched_frame = 0;
    uint64_t when_show_on_the_screen = 0;
} stitch_cost_times_t;

typedef struct Frame {
    int cam_id = 0;
    AVFrame* m_data = nullptr;
    stitch_cost_times_t m_costTimes;
    uint64_t m_timestamp = 0;
} stitch_frame_t;

struct Packet {
    int cam_id = 0;
    AVPacket* m_data = nullptr;
    stitch_cost_times_t m_costTimes;
    uint64_t m_timestamp = 0;
};

#ifdef __cplusplus
}
#endif
