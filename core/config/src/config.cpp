#include "config.h"

config::config() {
    loadFromFile("resource/hk5.json"); //需要修改，设置在一个文件夹下
}

bool config::loadFromFile(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Failed to open config file: " << filename << std::endl;
        return false;
    }

    json j;
    try {
        infile >> j;

        // 读取 global
        global.use_sub_input = j["global"]["use_sub_input"];
        global.software_status = j["global"]["software_status"];
        global.status = j["global"]["status"];
        global.save_rtsp_data_time = j["global"]["save_rtsp_data_time"];
        global.save_rtsp_data_path = j["global"]["save_rtsp_data_path"];

        // 读取 cameras
        for (const auto& cam : j["cameras"]) {
            CameraConfig c;
            c.name = cam["name"];
            c.input_url = cam["input_url"];
            c.sub_input_url = cam["sub_input_url"];
            c.output_url = cam["output_url"];

            // crop 是数组
            for (const auto& val : cam["crop"]) {
                c.crop.push_back(val);
            }

            c.rtsp = cam.value("rtsp", false); // 如果没有该字段则默认为 false

            // 解析 stitch
            if (cam.contains("stitch")) {
                const auto& s = cam["stitch"];
                c.stitch.enable = s.value("enable", false);
                c.stitch.mode = s.value("mode", "");
            }

            cameras.push_back(c);
        }

        // 读取 stitch
        stitch.output_url = j["stitch"]["output_url"];

        if(j["stitch"].contains("output_width")) {
            stitch.output_width = j["stitch"]["output_width"];
        } else {
            stitch.output_width = -1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        return false;
    }

    return true;
}

config& config::GetInstance() {  //是否线程安全？
    static config instance;
    return instance;
}

const GlobalConfig config::GetGlobalConfig() const {
    return global;
}

const std::vector<CameraConfig> config::GetCameraConfig() const {
    return cameras;
}

const GlobalStitchConfig config::GetGlobalStitchConfig() const {
    return stitch;
}