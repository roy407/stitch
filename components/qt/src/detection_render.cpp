#include "detection_render.h"
#include<QDebug>


detection_render::detection_render()
    : vbo(QOpenGLBuffer::VertexBuffer),
      detection_vbo(QOpenGLBuffer::VertexBuffer),
      idY(0),
      idUV(0),
      initialized(false)
{}
detection_render::~detection_render() {
    if (idY) glDeleteTextures(1, &idY);
    if (idUV) glDeleteTextures(1, &idUV);
}

bool detection_render::setupShaders() {
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

bool detection_render::setupGeometry() {
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

bool detection_render::setupTextures() {
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


bool detection_render::setupDetectionShaders() {
    const char *vsrc = R"(
        attribute vec4 vertexIn;
        void main() {
            gl_Position = vertexIn;
        }
    )";

    const char *fsrc = R"(
        uniform vec4 color;
        void main() {
            gl_FragColor = color;
        }
    )";

    if (!detection_program.addCacheableShaderFromSourceCode(QOpenGLShader::Vertex, vsrc) ||
        !detection_program.addCacheableShaderFromSourceCode(QOpenGLShader::Fragment, fsrc) ||
        !detection_program.link()) {
        qCritical() << "Detection Shader setup failed";
        return false;
    }
    return true;
}
bool detection_render::setupDetectionGeometry() {
    detection_vbo.create();
    if (!detection_vbo.isCreated()) {
        qCritical() << "Failed to create detection VBO";
        return false;
    }
    return true;
}
bool detection_render::initialize(){
    initializeOpenGLFunctions();
    bool frame_init=setupShaders()&&setupGeometry()&&setupTextures();
    bool detection_init=setupDetectionShaders()&&setupDetectionGeometry();
    initialized=frame_init&&detection_init;
    return initialized;         
}

void detection_render::render(uchar* nv12Ptr, int w, int h, int y_stride, int uv_stride) {
    if (!nv12Ptr || w <= 0 || h <= 0) {
        return;
    }
    glActiveTexture(GL_TEXTURE1);
    // 更新Y分量纹理
    glBindTexture(GL_TEXTURE_2D, idY);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, y_stride);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, nv12Ptr);

    // 更新UV分量纹理
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, idUV);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, uv_stride/2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG, w / 2, h / 2, 0, GL_RG, GL_UNSIGNED_BYTE,
                 nv12Ptr + y_stride * h);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    // 执行渲染
    executeGlRendering();
}
void detection_render::executeGlRendering(){
    program.bind();

    vbo.bind();

    program.setAttributeBuffer("vertexIn", GL_FLOAT, 0, 2, 4 * sizeof(GLfloat));
    program.enableAttributeArray("vertexIn");
    program.setAttributeBuffer("textureIn", GL_FLOAT, 2 * sizeof(GLfloat), 2, 4 * sizeof(GLfloat));
    program.enableAttributeArray("textureIn");

    program.setUniformValue("textureY", 1); // 纹理单元1绑定Y分量
    program.setUniformValue("textureUV", 0); // 纹理单元0

    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    program.release();

}
void detection_render:: renderDetections(const std::vector<detection_box>& detections,int img_width,int img_height){
   if(!initialized){
       return;
   }
   if(detections.empty()){
       return;
   }
   if(img_width<=0||img_height<=0){
       return;
   }
   glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    for(size_t i=0;i<detections.size();++i){
        const detection_box&box=detections[i];
        float x1=box.x*2.0f/img_width-1.0f;
        float y1=1.0f-box.y*2.0f/img_height;
         float x2 = (box.x + box.width) * 2.0f / img_width - 1.0f;  // 修正
        float y2 = 1.0f - (box.y + box.height) * 2.0f / img_height; 
        GLfloat vertices[] = {
            x1,  y1,  // 左上
            x2,  y1,  // 右上
            x2,  y2,  // 右下
            x1,  y2   // 左下
        };
          float r, g, b;
        if (box.confidence > 0.8f) {
            r = 0.0f; g = 1.0f; b = 0.0f;  // 绿色 - 高置信度
        } else if (box.confidence > 0.5f) {
            r = 1.0f; g = 1.0f; b = 0.0f;  // 黄色 - 中置信度
        } else {
            r = 1.0f; g = 0.0f; b = 0.0f;  // 红色 - 低置信度
        }
        
        // 渲染检测框边框
        renderDetectionBox(x1, y1, x2, y2, r, g, b, 1.0f);
    }
    glDisable(GL_BLEND);
}
void detection_render::renderDetectionBox(float x1, float y1, float x2, float y2,
                                         float r, float g, float b, float a) {
    // 定义矩形的4条边（8个顶点）
    // 每个顶点有4个分量：x, y, z, w
    float vertices[] = {
        // 上边
        x1, y1, 0.0f, 1.0f,  // 左上
        x2, y1, 0.0f, 1.0f,  // 右上
        
        // 右边
        x2, y1, 0.0f, 1.0f,  // 右上
        x2, y2, 0.0f, 1.0f,  // 右下
        
        // 下边
        x2, y2, 0.0f, 1.0f,  // 右下
        x1, y2, 0.0f, 1.0f,  // 左下
        
        // 左边
        x1, y2, 0.0f, 1.0f,  // 左下
        x1, y1, 0.0f, 1.0f   // 左上
    };
    
    // 绑定检测框着色器程序
    detection_program.bind();
    
    // 绑定检测框顶点缓冲区
    detection_vbo.bind();
    detection_vbo.allocate(vertices, sizeof(vertices));
    
    // 设置顶点属性
    detection_program.setAttributeBuffer("vertexIn", GL_FLOAT, 0, 4, 4 * sizeof(float));
    detection_program.enableAttributeArray("vertexIn");
    
    // 设置颜色
    detection_program.setUniformValue("drawColor", r, g, b, a);
    
    // 设置线宽
    glLineWidth(2.0f);
    
    // 绘制4条线
    glDrawArrays(GL_LINES, 0, 8);
    
    // 清理
    detection_program.release();
    detection_vbo.release();
}