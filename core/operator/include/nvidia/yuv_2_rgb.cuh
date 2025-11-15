#pragma once
#include <cstdint>

extern "C"
void convertNV12ToRGBA(uint8_t* d_y, uint8_t* d_uv, uchar4* d_rgba, int width, int height);

