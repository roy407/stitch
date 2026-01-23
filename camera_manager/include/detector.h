#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

struct DetectionResult {
    cv::Rect box;
    int class_id;
    float confidence;
};

class YoloOnnxDetector {
public:
    struct Options {
        Options();
        float score_threshold;
        float nms_threshold;
        bool enable_cuda;
    };

    YoloOnnxDetector(const std::string& model_path,
                     const std::string& labels_path,
                     const Options& options = Options());

    bool is_initialized() const;
    const std::vector<std::string>& labels() const;
    struct PreprocessedInput {
        cv::Mat letterboxed_image;
        float scale;
        int x_offset;
        int y_offset;
        int original_width;
        int original_height;
    };

    PreprocessedInput letterbox(const cv::Mat& image_bgr, cv::Mat& scratch) const;
    std::vector<DetectionResult> detectFromLetterbox(const PreprocessedInput& input);
    std::vector<DetectionResult> detect(const cv::Mat& image_bgr);
    float score_threshold() const { return options_.score_threshold; }
    float nms_threshold() const { return options_.nms_threshold; }

private:
    void load_labels(const std::string& labels_path);
    void prepare_io_info();
    void ensure_cuda_provider();

    static Ort::Env& global_env();

    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions session_options_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

    std::vector<std::string> labels_;
    Options options_;

    int input_height_ = 0;
    int input_width_ = 0;
    bool initialized_ = false;
};
