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

private:
    bool setupShaders();
    bool setupGeometry();
    bool setupTextures();
    void executeGlRendering();

    QOpenGLShaderProgram program;
    QOpenGLBuffer vbo;
    GLuint idY = 0, idUV = 0;
};

#endif // NV12RENDER_H