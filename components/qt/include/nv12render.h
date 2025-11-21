// nv12render.h
#ifndef NV12RENDER_H
#define NV12RENDER_H

#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>

class Nv12Render : protected QOpenGLFunctions {
public:
    Nv12Render();
    ~Nv12Render();

    bool initialize();
    void render(uchar* nv12Ptr, int w, int h, int y_stride, int uv_stride);
    void render(uchar* y_data, uchar* uv_data, int w, int h, int y_stride, int uv_stride);

private:
    bool setupShaders();
    bool setupGeometry();
    bool setupTextures();
    void executeGlRendering();

    QOpenGLShaderProgram program; //着色器程序
    QOpenGLBuffer vbo; //顶点缓冲区对象
    GLuint idY = 0, idUV = 0; //纹理ID
};

#endif // NV12RENDER_H