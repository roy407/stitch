#pragma once
#include "IStitch.h"
#include "nvidia_kernel.h"
#include "log.hpp"

struct StitchOps {
    using StitchFunc = AVFrame* (*)(void*, AVFrame**);
    using InitFunc   = void (*)(void*, int, int, int, int);

    void* obj = nullptr;  // 指向实际模板对象
    StitchFunc stitch = nullptr;
    InitFunc init = nullptr;
};

template<typename Impl>
StitchOps* make_stitch_ops(Impl* obj) {
    StitchOps* ops = new StitchOps;

    ops->obj = obj;

    ops->init = [](void* p, int num, int single_width, int output_width, int height) {
        static_cast<Impl*>(p)->init(num, single_width, output_width, height);
    };

    ops->stitch = [](void* p, AVFrame** inputs) -> AVFrame* {
        return static_cast<Impl*>(p)->do_stitch(inputs);
    };

    return ops;
}

template<typename Impl>
void delete_stitch_ops(StitchOps* ops) {
    delete static_cast<Impl*>(ops->obj);
    delete ops;
}

template <typename Format, typename KernelTag>
class StitchImpl
    : public IStitch<StitchImpl<Format, KernelTag>>, public KernelTag
{
public:
    void init_impl() 
    {
        if constexpr (Format::value == 0) {
            AllocateFrameBufPtrYUV420();
        } else if constexpr (Format::value == 1) {
            AllocateFrameBufPtrYUV420P();
        }

        if (!this->initgetKernelGpuMemory(this->num)) {
            LOG_ERROR("StitchImpl::init_impl failed: initgetKernelGpuMemory returned false (CUDA malloc/memcpy failed).");
            return;
        }

    }

    AVFrame* do_stitch_impl(AVFrame** inputs)
    {
        if constexpr (Format::value == 0) {
            MemoryCpyFrameBufPtrYUV420(inputs);
        } else if constexpr (Format::value == 1) {
            MemoryCpyFrameBufPtrYUV420P(inputs);
        }
        this->output = av_frame_alloc();
        this->output->format = AV_PIX_FMT_CUDA;
        this->output->width = this->output_width;
        this->output->height = this->height;
        this->output->hw_frames_ctx = av_buffer_ref(this->hw_frames_ctx);
        if (av_hwframe_get_buffer(this->hw_frames_ctx, this->output, 0) < 0) {
            throw std::runtime_error("Failed to allocate GPU AVFrame buffer");
        }
        launch_kernel();
        return this->output;
    }

    ~StitchImpl() {
        cudaFree(d_inputs_y);
        cudaFree(d_inputs_uv);
        cudaFree(d_input_linesize_y);
        cudaFree(d_input_linesize_uv);
        cudaFree(d_inputs_u);
        cudaFree(d_inputs_v);
        cudaFree(d_input_linesize_u);
        cudaFree(d_input_linesize_v);
        
        this->freeKernelGpuMemory();
    }

private:
    void launch_kernel();
private:
    uint8_t **d_inputs_y{nullptr};
    uint8_t **d_inputs_uv{nullptr};
    int* d_input_linesize_y{nullptr};
    int* d_input_linesize_uv{nullptr};
    bool AllocateFrameBufPtrYUV420();
    bool MemoryCpyFrameBufPtrYUV420(AVFrame** inputs);

    uint8_t **d_inputs_u{nullptr};
    uint8_t **d_inputs_v{nullptr};
    int* d_input_linesize_u{nullptr};
    int* d_input_linesize_v{nullptr};
    bool AllocateFrameBufPtrYUV420P();
    bool MemoryCpyFrameBufPtrYUV420P(AVFrame** inputs);
};

template <typename Format, typename KernelTag>
void StitchImpl<Format, KernelTag>::launch_kernel() {
    uint8_t* output_y = this->output->data[0];
    uint8_t* output_uv = this->output->data[1];

    if constexpr (Format::value == 0) {
        if constexpr (std::is_same_v<KernelTag, RawKernel>)
        {
            cudaStream_t stream = 0;
            launch_stitch_kernel_raw(
                d_inputs_y, d_inputs_uv,
                d_input_linesize_y, d_input_linesize_uv,
                output_y, output_uv,
                this->output->linesize[0], this->output->linesize[1],
                this->num, this->single_width, this->output_width, this->height,
                stream
            );
            cudaStreamSynchronize(stream);
        }
        else if constexpr (std::is_same_v<KernelTag, CropKernel>)
        {
            cudaStream_t stream = 0;
            launch_stitch_kernel_with_crop(
                d_inputs_y, d_inputs_uv,
                d_input_linesize_y, d_input_linesize_uv,
                output_y, output_uv,
                this->output->linesize[0], this->output->linesize[1],
                this->num, this->single_width, this->output_width, this->height,
                stream, this->d_crop
            );
            cudaStreamSynchronize(stream);
        }
        else if constexpr (std::is_same_v<KernelTag, HMatrixInvKernel>)
        {
            cudaStream_t stream = 0;
            launch_stitch_kernel_with_h_matrix_inv(
                d_inputs_y, d_inputs_uv,
                d_input_linesize_y, d_input_linesize_uv,
                output_y, output_uv,
                this->output->linesize[0], this->output->linesize[1],
                this->num, this->single_width, this->output_width, this->height,
                this->d_h_matrix_inv, this->d_cam_polygons, stream
            );
            cudaStreamSynchronize(stream);
        }
        else if constexpr (std::is_same_v<KernelTag, HMatrixInvV1_1Kernel>)
        {
            cudaStream_t stream1 = 0, stream2 = 0;
            launch_stitch_kernel_with_h_matrix_inv_v1_1(
                d_inputs_y, d_inputs_uv,
                d_input_linesize_y, d_input_linesize_uv,
                output_y, output_uv,
                this->output->linesize[0], this->output->linesize[1],
                this->num, this->single_width, this->output_width, this->height,
                this->d_h_matrix_inv, this->d_cam_polygons, stream1, stream2//作了修改
            );
            cudaStreamSynchronize(stream1);
            cudaStreamSynchronize(stream2);
        }
        else if constexpr (std::is_same_v<KernelTag, HMatrixInvV2Kernel>)
        {
            cudaStream_t stream1 = 0, stream2 = 0;
            launch_stitch_kernel_with_h_matrix_inv_v2(
                d_inputs_y, d_inputs_uv,
                d_input_linesize_y, d_input_linesize_uv,
                output_y, output_uv,
                this->output->linesize[0], this->output->linesize[1],
                this->num, this->single_width, this->output_width, this->height,
                this->d_h_matrix_inv, stream1, stream2
            );
            cudaStreamSynchronize(stream1);
            cudaStreamSynchronize(stream2);
        }
        else if constexpr (std::is_same_v<KernelTag, MappingTableKernel>)
        {
            cudaStream_t stream1 = 0, stream2 = 0;
            launch_stitch_kernel_with_mapping_table(
                d_inputs_y, d_inputs_uv,
                d_input_linesize_y, d_input_linesize_uv,
                output_y, output_uv,
                this->output->linesize[0], this->output->linesize[1],
                this->num, this->single_width, this->output_width, this->height,
                this->d_mapping_table, stream1, stream2
            );
            cudaStreamSynchronize(stream1);
            cudaStreamSynchronize(stream2);
        }
        else {
            static int __cnt__ = 0;
            __cnt__ ++;
            if(__cnt__ % 20 == 0) LOG_WARN("No kernel is running now");
        }
    } else if constexpr (Format::value == 1) {
        if constexpr (std::is_same_v<KernelTag, RawKernel>) {
            cudaStream_t stream = 0;
            launch_stitch_kernel_raw_yuv420p(
                d_inputs_y, d_inputs_u, d_inputs_v,
                d_input_linesize_y, d_input_linesize_u, d_input_linesize_v,
                output_y, output_uv,
                this->output->linesize[0], this->output->linesize[1],
                this->num, this->single_width, this->output_width, this->height,
                stream
            );
            cudaStreamSynchronize(stream);
        } else if constexpr (std::is_same_v<KernelTag, MappingTableKernel>) {
            cudaStream_t stream1 = 0, stream2 = 0;
            launch_stitch_kernel_with_mapping_table_yuv420p(
                d_inputs_y, d_inputs_u, d_inputs_v,
                d_input_linesize_y, d_input_linesize_u, d_input_linesize_v,
                output_y, output_uv,
                this->output->linesize[0], this->output->linesize[1],
                this->num, this->single_width, this->output_width, this->height,
                this->d_mapping_table, stream1, stream2
            );
            cudaStreamSynchronize(stream1);
            cudaStreamSynchronize(stream2);
        } else {
            static int __cnt__ = 0;
            __cnt__ ++;
            if(__cnt__ % 20 == 0) LOG_WARN("No kernel is running now");
        }
    }
}

template <typename Format, typename KernelTag>
bool StitchImpl<Format, KernelTag>::AllocateFrameBufPtrYUV420() {
    CHECK_CUDA(cudaMalloc(&d_inputs_y, sizeof(uint8_t*) * this->num));
    CHECK_CUDA(cudaMalloc(&d_inputs_uv, sizeof(uint8_t*) * this->num));
    CHECK_CUDA(cudaMalloc(&d_input_linesize_y, this->num * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_input_linesize_uv, this->num * sizeof(int)));
    return true;
}

template <typename Format, typename KernelTag>
bool StitchImpl<Format, KernelTag>::MemoryCpyFrameBufPtrYUV420(AVFrame** inputs) {
    uint8_t* gpu_inputs_y[this->num];
    uint8_t* gpu_inputs_uv[this->num];
    for (int i = 0; i < this->num; ++i) {
        if (!inputs[i]) {
            return false;
        }
        gpu_inputs_y[i] = inputs[i]->data[0];
        gpu_inputs_uv[i] = inputs[i]->data[1];
    }

    cudaMemcpy(d_inputs_y, gpu_inputs_y, sizeof(uint8_t*) * this->num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs_uv, gpu_inputs_uv, sizeof(uint8_t*) * this->num, cudaMemcpyHostToDevice);
    /*如果是不变量，考虑只初始化一次---待修改*/
    int h_input_linesize_y[this->num];
    int h_input_linesize_uv[this->num];

    for(int i=0;i<this->num;i++) {
        h_input_linesize_uv[i] = inputs[i]->linesize[1];
        h_input_linesize_y[i] = inputs[i]->linesize[0];
    }

    cudaMemcpy(d_input_linesize_y, h_input_linesize_y, this->num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_linesize_uv, h_input_linesize_uv, this->num * sizeof(int), cudaMemcpyHostToDevice);
    return true;
}

template <typename Format, typename KernelTag>
bool StitchImpl<Format, KernelTag>::AllocateFrameBufPtrYUV420P() {
    CHECK_CUDA(cudaMalloc(&d_inputs_y, sizeof(uint8_t*) * this->num));
    CHECK_CUDA(cudaMalloc(&d_inputs_u, sizeof(uint8_t*) * this->num));
    CHECK_CUDA(cudaMalloc(&d_inputs_v, sizeof(uint8_t*) * this->num));
    CHECK_CUDA(cudaMalloc(&d_input_linesize_y, this->num * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_input_linesize_u, this->num * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_input_linesize_v, this->num * sizeof(int)));
    return true;
}

template <typename Format, typename KernelTag>
bool StitchImpl<Format, KernelTag>::MemoryCpyFrameBufPtrYUV420P(AVFrame **inputs) {
    uint8_t* gpu_inputs_y[this->num];
    uint8_t* gpu_inputs_u[this->num];
    uint8_t* gpu_inputs_v[this->num];
    for (int i = 0; i < this->num; ++i) {
        if (!inputs[i]) {
            return false;
        }
        gpu_inputs_y[i] = inputs[i]->data[0];
        gpu_inputs_u[i] = inputs[i]->data[1];
        gpu_inputs_v[i] = inputs[i]->data[2];
    }

    cudaMemcpy(d_inputs_y, gpu_inputs_y, sizeof(uint8_t*) * this->num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs_u, gpu_inputs_u, sizeof(uint8_t*) * this->num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs_v, gpu_inputs_v, sizeof(uint8_t*) * this->num, cudaMemcpyHostToDevice);
    /*如果是不变量，考虑只初始化一次---待修改*/
    int h_input_linesize_y[this->num];
    int h_input_linesize_u[this->num];
    int h_input_linesize_v[this->num];

    for(int i=0;i<this->num;i++) {
        h_input_linesize_y[i] = inputs[i]->linesize[0];
        h_input_linesize_u[i] = inputs[i]->linesize[1];
        h_input_linesize_v[i] = inputs[i]->linesize[2];
    }

    cudaMemcpy(d_input_linesize_y, h_input_linesize_y, this->num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_linesize_u, h_input_linesize_u, this->num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_linesize_v, h_input_linesize_v, this->num * sizeof(int), cudaMemcpyHostToDevice);
    return true;
}


