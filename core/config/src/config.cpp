#include "config.h"
#include "log.hpp"

#undef CFG_HANDLE

std::string config::config_name = "";

config::config() {
    loadFromFile();
}

bool config::loadFromFile() {
    std::string filename = config_name;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cout<<"Failed to open config file: " <<filename<< std::endl;
        return false;
    }
    json j;
    infile >> j;
    if (j.contains("global")) loadGlobalConfig(j["global"], cfg.global);
    if (j.contains("pipeline")) {
        for (auto& p : j["pipeline"]) {
            PipelineConfig pipe;
            loadPipelineConfig(p, pipe);
            cfg.pipelines.push_back(pipe);
        }
    }
    return true;
}

void config::loadGlobalConfig(const json& j, GlobalConfig& cfg) {
    cfg.mode = j.value("mode", "debug");
    cfg.type = j.value("type", "mp4");
    cfg.format =j.value("format","YUV420");
    cfg.record_duration = j.value("record_duration", 240);
    cfg.record_path = j.value("record_path", "");
    cfg.decoder = j.value("decoder", "h264_cuvid");
    cfg.encoder = j.value("encoder", "h264_nvenc");
}

void config::loadCamerasInfo(std::string file_path, PipelineConfig& pipe) {
    std::ifstream infile(file_path);
    if (!infile.is_open()) {
        std::cout<<"Failed to open config file: " <<file_path<< std::endl;
        return;
    }
    json j;
    infile >> j;
    if(j.contains("cameras")) {
        for(auto& c : j["cameras"]) {
            CameraConfig cam;
            praseCameraConfig(c, cam);
            pipe.default_width += cam.width;
            pipe.default_height = cam.height;
            pipe.cameras.push_back(cam);
        }
    }
    if(j.contains("stitch")) {
        loadStitchConfig(j["stitch"], pipe.stitch, pipe.default_width, pipe.default_height);
    }
}

void config::loadStitchConfig(const json& j, StitchConfig& stitch, uint64_t default_width, uint64_t default_height) {
    if (j.contains("stitch_impl") && j["stitch_impl"].contains("mapping_table")) {
        auto m = j["stitch_impl"]["mapping_table"];
        stitch.stitch_impl.mapping_table.file_path = m.value("file_path", "");
        stitch.stitch_impl.mapping_table.output_width = m.value("output_width", -1);
        loadMappingTable(stitch.stitch_impl.mapping_table.d_mapping_table,
            stitch.stitch_impl.mapping_table.file_path, stitch.stitch_impl.mapping_table.output_width, default_height);
    }

    if (j.contains("stitch_impl") && j["stitch_impl"].contains("H_matrix_inv")) {
        auto h = j["stitch_impl"]["H_matrix_inv"];
        std::vector<std::array<double, 9>>& h_matrix_inv = stitch.stitch_impl.h_matrix_inv;
        for (auto& [key, mat] : h.items()) {
            std::array<double, 9> arr{};
            int idx = 0;
            for (const auto& row : mat) {
                for (const auto& val : row) {
                    arr[idx++] = val.get<double>();
                }
            }
            h_matrix_inv.push_back(arr);
        }
    }

    stitch.output_url = j.value("output_url", stitch.output_url);
}

void config::loadPipelineConfig(const json& j, PipelineConfig& pipe) {
    pipe.name = j.value("name", "");
    pipe.pipeline_id = j.value("pipeline_id", -1);
    pipe.enable = j.value("enable", false);
    if(j.contains("use_sub_input")) { // TODO：先暂时这么写，之后有问题继续修改
        pipe.use_substream = j.value("use_sub_input", false);
        pipe.main_stream = j.value("main_stream", "");
        pipe.sub_stream = j.value("sub_stream", "");
        if(pipe.use_substream == false) {
            loadCamerasInfo(pipe.main_stream, pipe);
        } else {
            loadCamerasInfo(pipe.sub_stream, pipe);
        }
        if(j.contains("stitch")) {
            pipe.stitch.stitch_mode = j["stitch"].value("stitch_mode", "raw");
        } else {
            pipe.stitch.stitch_mode = "raw";
        }
    } else {
        if(j.contains("cameras")) {
            for(auto& c : j["cameras"]) {
                CameraConfig cam;
                praseCameraConfig(c, cam);
                pipe.default_height = cam.height;
                pipe.default_width += cam.width;
                pipe.cameras.push_back(cam);
            }
        }
        if(j.contains("stitch")) {
            pipe.stitch.stitch_mode = j["stitch"].value("stitch_mode", "raw");
            loadStitchConfig(j["stitch"], pipe.stitch, pipe.default_width, pipe.default_height);
        } else {
            pipe.stitch.stitch_mode = "raw";
        }
    }
}

bool config::loadMappingTable(cudaTextureObject_t& tex,
                              const std::string filename,
                              uint64_t width,
                              uint64_t height)
{
    std::ifstream infile(filename, std::ios::binary);
    if (!infile.is_open()) {
        std::cout << "Failed to open mapping table: " << filename << std::endl;
        return false;
    }

    infile.seekg(0, std::ios::end);
    size_t file_bytes = infile.tellg();
    infile.seekg(0, std::ios::beg);

    size_t count = file_bytes / sizeof(uint16_t);
    std::vector<uint16_t> raw(count);
    infile.read((char*)raw.data(), file_bytes);

    size_t total = width * height;

    std::vector<MapEntry> buf(total);
    size_t idx = 0;
    for (uint64_t x = 0; x < width; x++) {
        for (uint64_t y = 0; y < height; y++) {
            size_t dst = y * width + x;

            buf[dst].cam_id = raw[idx++];
            buf[dst].map_x  = raw[idx++];
            buf[dst].map_y  = raw[idx++];
            buf[dst].pad    = 0;
        }
    }

    // 分配独立 cudaArray
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<ushort4>();
    cudaArray_t cuArray;
    CHECK_CUDA(cudaMallocArray(&cuArray, &desc, width, height));

    CHECK_CUDA(cudaMemcpy2DToArray(
        cuArray, 0, 0, buf.data(),
        width * sizeof(MapEntry),
        width * sizeof(MapEntry),
        height,
        cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc = {};
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.normalizedCoords = 0;

    CHECK_CUDA(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));

    return true;
}

void config::praseCameraConfig(const json & j, CameraConfig &cam) {
    cam.name = j.value("name", "");
    cam.cam_id = j.value("cam_id", -1);
    cam.enable = j.value("enable", false);
    cam.input_url = j.value("input_url", "");
    cam.width = j.value("width", -1);
    cam.height = j.value("height", -1);
    cam.enable_view = j.value("enable_view", false);
    cam.scale_factor = j.value("scale_factor", 1.0);
    cam.rtsp = j.value("rtsp", false);
    cam.output_url = j.value("output_url", "");
}

void config::SetConfigFileName(std::string cfg_name) {
    config_name = cfg_name;
}

config &config::GetInstance()
{ // 是否线程安全？
    static config instance;
    return instance;
}

const Config config::GetConfig() const {
    return cfg;
}

const GlobalConfig config::GetGlobalConfig() const {
    return cfg.global;
}

const PipelineConfig config::GetPipelineConfig(int pipeline_id) const {
    if(pipeline_id >= cfg.pipelines.size() || pipeline_id < 0) {
        throw std::runtime_error("Failed to pipeline id: " + std::to_string(pipeline_id));
    }
    return cfg.pipelines[pipeline_id];
}

const std::vector<CameraConfig> config::GetCamerasConfig(int pipeline_id) const {
    if(pipeline_id >= cfg.pipelines.size() || pipeline_id < 0) {
        throw std::runtime_error("Failed to pipeline id: " + std::to_string(pipeline_id));
    }
    return cfg.pipelines[pipeline_id].cameras;
}

const StitchConfig config::GetStitchConfig(int pipeline_id) const {
    if(pipeline_id >= cfg.pipelines.size() || pipeline_id < 0) {
        throw std::runtime_error("Failed to pipeline id: " + std::to_string(pipeline_id));
    }
    return cfg.pipelines[pipeline_id].stitch;
}
