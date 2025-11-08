#include <iostream> 
#include <cmath>
#include "config.h"
void applyHomography(double* H, float x, float y, float* out_x, float* out_y);
bool is_point_in_quadrilateral(float x, float y, float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4);

int main() {
    config& cfg = config::GetInstance();
    auto stitch_cfg = cfg.GetGlobalStitchConfig();
    std::vector<std::array<double, 9>> h_matrix = stitch_cfg.h_matrix;
    std::vector<std::array<float, 8>> cam_polygons = stitch_cfg.cam_polygons;
    int width = 3840;  // 拼接前单个图像宽度
    int height = 2160; // 拼接前单个图像高度
    int cam_num = h_matrix.size();
    int offset = -6339; // 由于在求h矩阵时，以中间的图为平面，会导致左边的像素点位置都是负数，因此需要一个偏移值

    std::string filename = "hk5.bin";
    std::ofstream fout(filename, std::ios::binary);
    if (!fout.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    for(int x=0;x<width * cam_num;x++) {
        for(int y=0;y<height;y++) {
            int active_cam = -1;
            float out_x = -1;
            float out_y = -1;
            for (int cam = cam_num-1; cam >= 0; --cam) {
                float* quad = cam_polygons[cam].data();
                if (is_point_in_quadrilateral(x + offset, y, 
                    quad[0], quad[1], quad[2], quad[3], 
                    quad[4], quad[5], quad[6], quad[7])) {
                    active_cam = cam;
                    applyHomography(h_matrix[active_cam].data(), x + offset,y, &out_x, &out_y);
                    break;
                }
            }
            // 如果没有匹配到相机，则写入默认值 -1
            unsigned short cam_id  = static_cast<unsigned short>(active_cam); // active_cam为-1时，代表没有匹配到任何一个位置
            unsigned short map_x   = static_cast<unsigned short>(out_x >= 0 ? out_x : 0);
            unsigned short map_y   = static_cast<unsigned short>(out_y >= 0 ? out_y : 0);

            // 写入三个 unsigned short（6字节）
            fout.write(reinterpret_cast<char*>(&cam_id), sizeof(unsigned short));
            fout.write(reinterpret_cast<char*>(&map_x), sizeof(unsigned short));
            fout.write(reinterpret_cast<char*>(&map_y), sizeof(unsigned short));
        }
    }
}

void applyHomography(double* H, float x, float y, float* out_x, float* out_y) {
    float denominator = H[6]*x + H[7]*y + H[8];
    if (fabsf(denominator) < 1e-6f) {
        *out_x = -1;
        *out_y = -1;
        return;
    }
    *out_x = (H[0]*x + H[1]*y + H[2]) / denominator;
    *out_y = (H[3]*x + H[4]*y + H[5]) / denominator;
}

bool is_point_in_quadrilateral(float x, float y,
    float x1, float y1, float x2, float y2,
    float x3, float y3, float x4, float y4)
{
    // 向量叉积法判断点是否在凸四边形内
    auto cross = [](float ax, float ay, float bx, float by) {
        return ax * by - ay * bx;
    };

    float d1 = cross(x - x1, y - y1, x2 - x1, y2 - y1);
    float d2 = cross(x - x2, y - y2, x3 - x2, y3 - y2);
    float d3 = cross(x - x3, y - y3, x4 - x3, y4 - y3);
    float d4 = cross(x - x4, y - y4, x1 - x4, y1 - y4);

    return (d1 * d3 >= 0) && (d2 * d4 >= 0);
}