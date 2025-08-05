#include "Stitch.h"
#include "operator_ascend.h"
#include <iostream>

#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixfmt.h>
#include <acl/acl.h>

#include "cuda_handle_init.h"
#include "config.h"

Stitch::Stitch(int width,int height,int cam_num): single_width(width),height(height),cam_num(cam_num) {
    
    output_width = single_width * cam_num;
    int* crop = new int[cam_num * 4];
    memset(crop,0,cam_num * 4);
    const std::vector<CameraConfig> cams = config::GetInstance().GetCameraConfig();
    for(int i=0;i<cams.size();i++) {
        if(cams[i].stitch.enable == true) {
            if(cams[i].stitch.mode == "crop") {
                std::vector<float> __crop = cams[i].crop;
                crop[i*4] = __crop[0] * width;
                crop[i*4+1] = __crop[1] * height;
                crop[i*4+2] = __crop[2] * width;
                crop[i*4+3] = __crop[3] * height;
            }
        }
    }
    output_width = 16251;
    //TODO：已经计算出了裁剪区域，应该可以直接得到输出图像的大小
    aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST;
    aclError ret;
    ret = aclrtMalloc((void**)&d_h_matrices, sizeof(float) * 9 * cam_num, policy);
    if (ret != ACL_ERROR_NONE) {
    }
    ret = aclrtMalloc((void**)&d_inputs_y, sizeof(uint8_t*) * cam_num, policy);
    if (ret != ACL_ERROR_NONE) {
    }
    ret = aclrtMalloc((void**)&d_inputs_uv, sizeof(uint8_t*) * cam_num, policy);
    if (ret != ACL_ERROR_NONE) {
    }
    ret = aclrtMalloc((void**)&d_crop, cam_num * 4 * sizeof(int), policy);
    if (ret != ACL_ERROR_NONE) {
    }
    ret = aclrtMemcpy(d_crop, cam_num * sizeof(int), crop, cam_num * sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
    }
    ret = aclrtMalloc((void**)&d_input_linesize_y, cam_num * sizeof(int), policy);
    if (ret != ACL_ERROR_NONE) {
    }
    ret = aclrtMalloc((void**)&d_input_linesize_uv, cam_num * sizeof(int), policy);
    if (ret != ACL_ERROR_NONE) {
    }
    // 创建 HW frame context
    hw_frames_ctx = av_hwframe_ctx_alloc(cuda_handle_init::GetGPUDeviceHandle());
    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)hw_frames_ctx->data;
    frames_ctx->format = AV_PIX_FMT_ASCEND;
    frames_ctx->sw_format = AV_PIX_FMT_NV12;   // CUDA 支持的底层格式
    frames_ctx->width = output_width;
    frames_ctx->height = height;
    frames_ctx->initial_pool_size = 20;

    if (av_hwframe_ctx_init(hw_frames_ctx) < 0) {
        throw std::runtime_error("Failed to initialize CUDA hwframe context");
    }
    delete[] crop;
    running.store(true);
}

Stitch::~Stitch() {
    running.store(false);
    // 释放设备内存（不检查返回值，默认成功）
    (void)aclrtFree(d_inputs_y);
    (void)aclrtFree(d_inputs_uv);
    (void)aclrtFree(d_crop);
    (void)aclrtFree(d_input_linesize_y);
    (void)aclrtFree(d_input_linesize_uv);
    (void)aclrtFree(d_h_matrices);
}

AVFrame* Stitch::do_stitch(AVFrame** inputs) {

    static float h_matrices[5][9] = {
        // 相机1到相机1的变换（单位矩阵）
        {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
        {9.99904053e-01,-4.19696324e-02,-3.38373637e+03,-6.66393412e-03,9.90487514e-01,8.77086171e+00,-3.82220590e-06,-3.81567906e-06,1.01298991e+00},
        {1.01286684e+00,-4.95562446e-02,-6.88883065e+03,-2.96245676e-02,9.81192353e-01,7.96030310e+01,-6.24623565e-06,-8.57136710e-06,1.03008289e+00},
        {1.02302896e+00,-5.09373902e-02,-1.03834207e+04,-4.23372572e-02,9.59948821e-01,3.90253967e+02,-1.52826843e-05,-1.50554885e-05,1.12208562e+00},
        {1.06537166e+00,-3.55742112e-02,-1.41281088e+04,-4.50705833e-02,9.45190099e-01,4.48848430e+02,-2.93221140e-05,-1.86384708e-05,1.30890439e+00}
    };

    uint8_t* gpu_inputs_y[cam_num];
    uint8_t* gpu_inputs_uv[cam_num];
    for (int i = 0; i < cam_num; ++i) {
        if (!inputs[i]) {
            return nullptr;
        }
        gpu_inputs_y[i] = inputs[i]->data[0];
        gpu_inputs_uv[i] = inputs[i]->data[1];
    }

    output = av_frame_alloc();
    if (!output) {
        throw std::runtime_error("Failed to allocate output frame");
    }

    output->format = AV_PIX_FMT_ASCEND;
    output->width = output_width;
    output->height = height;

    output->hw_frames_ctx = av_buffer_ref(hw_frames_ctx);

    if (av_hwframe_get_buffer(hw_frames_ctx, output, 0) < 0) {
        throw std::runtime_error("Failed to allocate GPU AVFrame buffer");
    }

    (void)aclrtMemcpy(d_h_matrices, sizeof(float)*9*cam_num, h_matrices, sizeof(float)*9*cam_num, ACL_MEMCPY_HOST_TO_DEVICE);
    (void)aclrtMemcpy(d_inputs_y, sizeof(uint8_t*)*cam_num, gpu_inputs_y, sizeof(uint8_t*)*cam_num, ACL_MEMCPY_HOST_TO_DEVICE);
    (void)aclrtMemcpy(d_inputs_uv, sizeof(uint8_t*)*cam_num, gpu_inputs_uv, sizeof(uint8_t*)*cam_num, ACL_MEMCPY_HOST_TO_DEVICE);

    uint8_t* output_y = output->data[0];
    uint8_t* output_uv = output->data[1];

    aclrtStream stream = nullptr;
    aclError ret = aclrtCreateStream(&stream);
    if (ret != ACL_ERROR_NONE) {
        printf("Failed to create ACL stream, error code: %d\n", ret);
        return nullptr;
    }

    /*如果是不变量，考虑只初始化一次---待修改*/
    int h_input_linesize_y[cam_num];
    int h_input_linesize_uv[cam_num];

    for(int i=0;i<cam_num;i++) {
        h_input_linesize_uv[i] = inputs[i]->linesize[1];
        h_input_linesize_y[i] = inputs[i]->linesize[0];
    }

    (void)aclrtMemcpy(d_input_linesize_y, cam_num * sizeof(int), h_input_linesize_y, cam_num * sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE);
    (void)aclrtMemcpy(d_input_linesize_uv, cam_num * sizeof(int), h_input_linesize_uv, cam_num * sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE);

    launch_stitch_kernel_raw(d_inputs_y, d_inputs_uv,
                        d_input_linesize_y, d_input_linesize_uv,
                        output_y, output_uv,
                        output->linesize[0], output->linesize[1],
                        cam_num, single_width, output_width, height,
                        stream);
    return output;
}
