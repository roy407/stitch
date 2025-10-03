// nv12render.cpp
#include "nv12render.h"
#include <QOpenGLTexture>
#include <QDebug>

Nv12Render::Nv12Render() {}
Nv12Render::~Nv12Render() {
    if (idY) glDeleteTextures(1, &idY);
    if (idUV) glDeleteTextures(1, &idUV);
}

bool Nv12Render::setupShaders() {
    const char *vsrc = R"(
        attribute vec4 vertexIn;
        attribute vec4 textureIn;
        varying vec4 textureOut;
        void main() {
            gl_Position = vertexIn;
            textureOut = textureIn;
        }
    )";

    const char *fsrc = R"(
        varying mediump vec4 textureOut;
        uniform sampler2D textureY;
        uniform sampler2D textureUV;
        void main() {
            vec3 yuv;
            yuv.x = texture2D(textureY, textureOut.st).r - 0.0625;
            yuv.y = texture2D(textureUV, textureOut.st).r - 0.5;
            yuv.z = texture2D(textureUV, textureOut.st).g - 0.5;
            vec3 rgb = mat3(1, 1, 1, 0, -0.39465, 2.03211, 1.13983, -0.58060, 0) * yuv;
            gl_FragColor = vec4(rgb, 1);
        }
    )";

    if (!program.addCacheableShaderFromSourceCode(QOpenGLShader::Vertex, vsrc) ||
        !program.addCacheableShaderFromSourceCode(QOpenGLShader::Fragment, fsrc) ||
        !program.link()) {
        qCritical() << "Shader setup failed";
        return false;
    }
    return true;
}

bool Nv12Render::setupGeometry() {
    GLfloat vertices[] = {
        -1.0f,  1.0f, 0.0f, 0.0f,  // 左上
         1.0f,  1.0f, 1.0f, 0.0f,  // 右上
         1.0f, -1.0f, 1.0f, 1.0f,  // 右下
        -1.0f, -1.0f, 0.0f, 1.0f   // 左下
    };

    vbo.create();
    vbo.bind();
    vbo.allocate(vertices, sizeof(vertices));
    return true;
}

bool Nv12Render::setupTextures() {
    glGenTextures(1, &idY);
    glBindTexture(GL_TEXTURE_2D, idY);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenTextures(1, &idUV);
    glBindTexture(GL_TEXTURE_2D, idUV);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    return true;
}

bool Nv12Render::initialize() {
    initializeOpenGLFunctions();
    return setupShaders() && setupGeometry() && setupTextures();
}

void Nv12Render::render(uchar* nv12Ptr, int w, int h, int y_stride, int uv_stride) {
    if (!nv12Ptr || w <= 0 || h <= 0) return;

    // 上传Y数据
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, idY);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, y_stride);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, nv12Ptr);

    // 上传UV数据
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, idUV);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, uv_stride / 2); // UV stride通常是Y stride的一半
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG, w/2, h/2, 0, GL_RG, GL_UNSIGNED_BYTE, nv12Ptr + y_stride * h);

    // 重置行对齐
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

    executeGlRendering();
}

void Nv12Render::executeGlRendering() {
    program.bind();
    vbo.bind();

    program.setAttributeBuffer("vertexIn", GL_FLOAT, 0, 2, 4 * sizeof(GLfloat));
    program.setAttributeBuffer("textureIn", GL_FLOAT, 2 * sizeof(GLfloat), 2, 4 * sizeof(GLfloat));
    program.enableAttributeArray("vertexIn");
    program.enableAttributeArray("textureIn");

    program.setUniformValue("textureUV", 0);
    program.setUniformValue("textureY", 1);

    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    program.release();
}