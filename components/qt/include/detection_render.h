#ifndef DETECTION_RENDER_H
#define DETECTION_RENDER_H

#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>  
#include <QMatrix4x4>
#include <vector>
#include <string>
#include <QOpenGLFunctions>
struct detection_box {
    float x; // 左上角x坐标
    float y; // 左上角y坐标
    float width; // 宽度
    float height; // 高度
    float confidence; // 置信度
    std::string label; // 标签
};
class detection_render : protected QOpenGLFunctions {
public:
    detection_render();
    ~detection_render();

    bool initialize();
    void render(uchar* nv12Ptr, int w, int h, int y_stride, int uv_stride);
    void renderDetections(const std::vector<detection_box>& boxes,int img_width, int img_height);

private:
  bool setupShaders();
    bool setupGeometry();
    bool setupTextures();
    void executeGlRendering();

    bool setupDetectionShaders();
    bool setupDetectionGeometry();
    void renderDetectionBox(float x1,float x2,float y1,float y2,float r,float g,float b,float a);
private:
    QOpenGLShaderProgram program;
    QOpenGLBuffer vbo;
    GLuint idY;                      // Y分量纹理ID
    GLuint idUV;     

    QOpenGLShaderProgram detection_program;
    QOpenGLBuffer detection_vbo;             // 检测框顶点缓冲区

    bool initialized;
};

#endif // DETECTION_RENDER_H