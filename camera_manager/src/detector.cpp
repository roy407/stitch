#include "detector.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <onnxruntime_cxx_api.h>

namespace {
constexpr const char* kLoggerName = "people_detector";

Ort::Env& make_env_once() {
    static Ort::Env env{ORT_LOGGING_LEVEL_WARNING, kLoggerName};
    return env;
}
} // namespace

YoloOnnxDetector::Options::Options()
    : score_threshold(0.25f),
      nms_threshold(0.45f),
      enable_cuda(true) {}

YoloOnnxDetector::YoloOnnxDetector(const std::string& model_path,
                                   const std::string& labels_path,
                                   const Options& options)
    : options_(options) {
    load_labels(labels_path);

    try {
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        ensure_cuda_provider();
        session_ = std::make_unique<Ort::Session>(global_env(), model_path.c_str(), session_options_);
        prepare_io_info();
        initialized_ = true;
    } catch (const Ort::Exception&) {
        initialized_ = false;
        session_.reset();
    }
}

bool YoloOnnxDetector::is_initialized() const {
    return initialized_;
}

const std::vector<std::string>& YoloOnnxDetector::labels() const {
    return labels_;
}

std::vector<DetectionResult> YoloOnnxDetector::detect(const cv::Mat& image_bgr) {
    cv::Mat scratch;
    auto preprocessed = letterbox(image_bgr, scratch);
    return detectFromLetterbox(preprocessed);
}

YoloOnnxDetector::PreprocessedInput YoloOnnxDetector::letterbox(const cv::Mat& image_bgr, cv::Mat& scratch) const {
    PreprocessedInput prep;
    if (image_bgr.empty() || input_width_ <= 0 || input_height_ <= 0) {
        return prep;
    }

    const float width_scale = static_cast<float>(input_width_) / static_cast<float>(image_bgr.cols);
    const float height_scale = static_cast<float>(input_height_) / static_cast<float>(image_bgr.rows);
    prep.scale = std::min(width_scale, height_scale);
    prep.original_width = image_bgr.cols;
    prep.original_height = image_bgr.rows;

    int resized_w = std::max(1, static_cast<int>(std::round(image_bgr.cols * prep.scale)));
    int resized_h = std::max(1, static_cast<int>(std::round(image_bgr.rows * prep.scale)));

    cv::Mat resized;
    cv::resize(image_bgr, resized, cv::Size(resized_w, resized_h));

    scratch.create(input_height_, input_width_, CV_8UC3);
    scratch.setTo(cv::Scalar(114, 114, 114));
    prep.x_offset = (input_width_ - resized_w) / 2;
    prep.y_offset = (input_height_ - resized_h) / 2;
    resized.copyTo(scratch(cv::Rect(prep.x_offset, prep.y_offset, resized_w, resized_h)));
    prep.letterboxed_image = scratch;

    return prep;
}

std::vector<DetectionResult> YoloOnnxDetector::detectFromLetterbox(const PreprocessedInput& input) {
    std::vector<DetectionResult> results;
    if (!initialized_ || input.letterboxed_image.empty()) {
        return results;
    }

    cv::Mat blob = cv::dnn::blobFromImage(input.letterboxed_image, 1.0f / 255.0f,
                                          cv::Size(input_width_, input_height_),
                                          cv::Scalar(), true, false, CV_32F);

    std::array<int64_t, 4> input_shape{1, 3, input_height_, input_width_};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                              blob.ptr<float>(),
                                                              blob.total(),
                                                              input_shape.data(),
                                                              input_shape.size());

    std::vector<const char*> input_names_cstr;
    std::vector<const char*> output_names_cstr;
    input_names_cstr.reserve(input_names_.size());
    output_names_cstr.reserve(output_names_.size());
    for (const auto& name : input_names_) {
        input_names_cstr.push_back(name.c_str());
    }
    for (const auto& name : output_names_) {
        output_names_cstr.push_back(name.c_str());
    }

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                        input_names_cstr.data(), &input_tensor, 1,
                                        output_names_cstr.data(), output_names_cstr.size());

    if (output_tensors.empty()) {
        return results;
    }

    const Ort::Value& first_output = output_tensors.front();
    auto output_info = first_output.GetTensorTypeAndShapeInfo();
    auto output_shape = output_info.GetShape();

    if (output_shape.size() < 3) {
        return results;
    }

    int64_t num_classes = output_shape[1] - 4;
    int64_t num_predictions = output_shape[2];

    const float* output_data = first_output.GetTensorData<float>();
    cv::Mat output_mat(output_shape[1], output_shape[2], CV_32F, const_cast<float*>(output_data));
    cv::Mat transposed = output_mat.t();

    const float inv_scale = input.scale > 0.0f ? 1.0f / input.scale : 1.0f;
    const float x_offset_f = static_cast<float>(input.x_offset);
    const float y_offset_f = static_cast<float>(input.y_offset);

    auto clamp_rect = [&](cv::Rect& rect) {
        rect.x = std::clamp(rect.x, 0, input.original_width - 1);
        rect.y = std::clamp(rect.y, 0, input.original_height - 1);
        int x2 = std::clamp(rect.x + rect.width, 0, input.original_width);
        int y2 = std::clamp(rect.y + rect.height, 0, input.original_height);
        rect.width = std::max(0, x2 - rect.x);
        rect.height = std::max(0, y2 - rect.y);
    };

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (int i = 0; i < transposed.rows; ++i) {
        const float* row_ptr = transposed.ptr<float>(i);
        cv::Mat scores(1, num_classes, CV_32F, const_cast<float*>(row_ptr + 4));
        cv::Point class_id_point;
        double max_class_score;
        cv::minMaxLoc(scores, nullptr, &max_class_score, nullptr, &class_id_point);

        float confidence = static_cast<float>(max_class_score);
        if (confidence < options_.score_threshold) {
            continue;
        }

        float cx = row_ptr[0];
        float cy = row_ptr[1];
        float w = row_ptr[2];
        float h = row_ptr[3];

        float left_f = (cx - 0.5f * w - x_offset_f) * inv_scale;
        float top_f = (cy - 0.5f * h - y_offset_f) * inv_scale;
        float width_f = w * inv_scale;
        float height_f = h * inv_scale;

        cv::Rect box(
            static_cast<int>(std::round(left_f)),
            static_cast<int>(std::round(top_f)),
            static_cast<int>(std::round(width_f)),
            static_cast<int>(std::round(height_f))
        );
        clamp_rect(box);
        if (box.area() <= 0) {
            continue;
        }

        boxes.push_back(box);
        confidences.push_back(confidence);
        class_ids.push_back(class_id_point.x);
    }

    std::vector<int> kept;
    cv::dnn::NMSBoxes(boxes, confidences, options_.score_threshold,
                      options_.nms_threshold, kept);

    results.reserve(kept.size());
    for (int idx : kept) {
        results.push_back(DetectionResult{
            boxes[idx],
            class_ids[idx],
            confidences[idx]
        });
    }

    return results;
}

void YoloOnnxDetector::load_labels(const std::string& labels_path) {
    labels_.clear();
    std::ifstream ifs(labels_path);
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty()) {
            labels_.push_back(line);
        }
    }
}

void YoloOnnxDetector::prepare_io_info() {
    input_names_.clear();
    output_names_.clear();

    size_t input_count = session_->GetInputCount();
    size_t output_count = session_->GetOutputCount();

    input_names_.reserve(input_count);
    output_names_.reserve(output_count);

    constexpr size_t kMaxTensorDims = 8;
    for (size_t i = 0; i < input_count; ++i) {
        auto name = session_->GetInputNameAllocated(i, allocator_);
        input_names_.push_back(std::string(name.get()));

        auto type_info = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        size_t dim_count = type_info.GetDimensionsCount();
        if (dim_count == 0 || dim_count > kMaxTensorDims) {
            continue;
        }

        std::vector<int64_t> shape = type_info.GetShape();
        if (shape.size() == 4) {
            int h = static_cast<int>(shape[2]);
            int w = static_cast<int>(shape[3]);
            if (h <= 0) h = 640;
            if (w <= 0) w = 640;
            input_height_ = h;
            input_width_ = w;
        }
    }

    for (size_t i = 0; i < output_count; ++i) {
        auto name = session_->GetOutputNameAllocated(i, allocator_);
        output_names_.push_back(std::string(name.get()));
    }
}

void YoloOnnxDetector::ensure_cuda_provider() {
    if (!options_.enable_cuda) {
        return;
    }

    OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options_, 0);
    if (status != nullptr) {
        Ort::GetApi().ReleaseStatus(status);
    }
}

Ort::Env& YoloOnnxDetector::global_env() {
    return make_env_once();
}


