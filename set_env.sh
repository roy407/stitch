#!/bin/bash

set -e

echo "=== 多媒体处理环境一键安装脚本 ==="
echo "适用于: Ubuntu 20.04/22.04"
echo "安装内容: CUDA, Qt, FFmpeg, OpenCV, Python环境"
echo "预计时间: 30-60分钟"
echo ""

# 检查是否在Docker中
if [ -f /.dockerenv ]; then
    echo "检测到Docker环境"
    IS_DOCKER=true
else
    echo "检测到物理机环境"
    IS_DOCKER=false
fi

# === 强制使用国内镜像源，避免网络超时 ===
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_DEFAULT_TIMEOUT="100"
export PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
# ======================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# FFmpeg 缓存配置
FFMPEG_VERSION="6.1.1"
FFMPEG_CACHE_DIR="/opt/ffmpeg-cache"
FFMPEG_TAR=" $ FFMPEG_CACHE_DIR/ffmpeg- $ FFMPEG_VERSION.tar.gz"

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 检查root权限
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}请使用sudo运行此脚本: sudo bash $0${NC}"
    exit 1
fi

# 检查操作系统
if [ ! -f /etc/os-release ]; then
    log_error "无法检测操作系统"
    exit 1
fi

source /etc/os-release
if [ "$ID" != "ubuntu" ]; then
    log_error "本脚本仅支持Ubuntu系统"
    exit 1
fi

log_info "检测到: $PRETTY_NAME"

# 确认安装
echo ""
echo "这个脚本将安装以下内容："
echo "1. NVIDIA CUDA Toolkit 11.8（如果未安装）"
echo "2. Qt 5 开发环境"
echo "3. FFmpeg 6.1.1（支持硬件加速）"
echo "4. OpenCV 和其他图像处理库"
echo "5. Python 3 数据科学工具包"
echo "6. 各种开发依赖库"
echo ""
echo "需要稳定的网络连接和大约 5GB 磁盘空间。"
echo ""

read -p "是否继续安装？[Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    log_info "安装取消"
    exit 0
fi

# ==============================
# 安装步骤函数
# ==============================

# 1. 更新系统和安装基础工具
update_system() {
    log_info "更新系统包列表..."
    apt-get update
    
    log_info "升级现有软件包..."
    apt-get upgrade -y
    
    log_info "安装基础工具..."
    apt-get install -y \
        software-properties-common \
        build-essential \
        wget \
        curl \
        git \
        ca-certificates \
        apt-transport-https \
        gnupg2 \
        lsb-release
    
    log_success "系统更新完成"
}

# 2. 安装CUDA（如果未安装）
install_cuda() {
    log_info "检查CUDA安装状态..."
    
    # 检查是否已安装CUDA
    if command -v nvcc &> /dev/null; then
        log_success "CUDA已安装:"
        nvcc --version | head -n 4
        return 0
    fi
    
    # 检查CUDA目录
    if [ -d "/usr/local/cuda" ]; then
        log_success "找到CUDA目录: /usr/local/cuda"
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /etc/profile
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /etc/profile
        #source /etc/profile
        return 0
    fi
    
    # 询问是否安装CUDA
    echo ""
    echo "未检测到CUDA。请到英伟达官网安装。"
    return 0
}

check_and_fix_cuda() {
    log_info "检查并修复CUDA环境..."

    # 查找CUDA安装路径
    CUDA_PATH=""
    if [ -d "/usr/local/cuda-11.8" ]; then
        CUDA_PATH="/usr/local/cuda-11.8"
    elif [ -d "/usr/local/cuda" ]; then
        CUDA_PATH="/usr/local/cuda"
    fi

    if [ -z "$CUDA_PATH" ]; then
        log_warning "未找到CUDA安装路径"
        return 1
    fi

    log_info "找到CUDA路径: $CUDA_PATH"

    # 设置环境变量
    export PATH="$CUDA_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
    export CUDA_HOME="$CUDA_PATH"

    # 验证nvcc
    if command -v nvcc &> /dev/null; then
        log_success "nvcc可用: $(which nvcc)"
        return 0
    else
        log_error "nvcc不可用，请检查CUDA安装"

        # 尝试创建符号链接
        if [ -x "$CUDA_PATH/bin/nvcc" ]; then
            log_info "尝试创建nvcc符号链接..."
            ln -sf "$CUDA_PATH/bin/nvcc" /usr/local/bin/nvcc 2>/dev/null || true
            export PATH="/usr/local/bin:$PATH"

            if command -v nvcc &> /dev/null; then
                log_success "通过符号链接nvcc可用"
                return 0
            fi
        fi

        return 1
    fi
}

# 3. 安装所有系统依赖 - 与Dockerfile完全一致
install_all_dependencies() {
    log_info "安装所有系统依赖包..."
    
    apt-get update
    
    # 安装基础开发工具（与Dockerfile完全一致）
    apt-get install -y \
        build-essential \
        wget \
        curl \
        git \
        pkg-config \
        cmake \
        nasm \
        yasm \
        autoconf \
        automake \
        libtool \
        libspdlog-dev
    
    # 安装Qt环境（与Dockerfile完全一致）
    apt-get install -y \
        qt5-default \
        qtbase5-dev \
        libqt5opengl5-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev
    
    # 安装系统FFmpeg开发包（与Dockerfile完全一致）
    apt-get install -y \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
        libavfilter-dev
    
    # 安装项目额外依赖（与Dockerfile完全一致）
    apt-get install -y \
        nlohmann-json3-dev \
        libopencv-dev \
        libboost-all-dev \
        python3 \
        python3-pip \
        python3-dev \
        python3-numpy \
        python3-matplotlib \
        python3-pandas \
        libeigen3-dev \
        libtbb-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libcurl4-openssl-dev \
        libssl-dev
    
    # 安装FFmpeg编译依赖（与Dockerfile完全一致）
    apt-get install -y \
        libx264-dev \
        libx265-dev \
        libvpx-dev \
        libmp3lame-dev \
        libopus-dev \
        libfdk-aac-dev \
        libwebp-dev \
        libfreetype6-dev \
        libfontconfig1-dev \
        libass-dev \
        libsdl2-dev \
        libva-dev \
        libvdpau-dev \
        libdrm-dev \
        libxml2-dev \
        libx11-dev \
        libxcb-shm0-dev \
        libxcb-xfixes0-dev
    
    # 安装NVIDIA NPP库（与Dockerfile完全一致）
    if command -v nvcc &> /dev/null || [ -d "/usr/local/cuda" ]; then
        apt-get install -y libnpp-dev-11-8 2>/dev/null || \
        apt-get install -y libnpp-dev 2>/dev/null || \
        log_warning "无法安装libnpp-dev，将尝试其他方式"
    fi
    
    log_success "所有系统依赖安装完成"
}

# 安装特定版本的NVIDIA Video Codec SDK
install_nvidia_codec_sdk() {
    log_info "安装NVIDIA Video Codec SDK..."
    
    WORKDIR="/tmp/nvidia-codec-$(date +%s)"
    mkdir -p "$WORKDIR"
    cd "$WORKDIR"
    
    # 方法1：尝试下载预编译的版本
    log_info "下载预编译的NVIDIA Video Codec SDK..."
    
    # 尝试多个版本
    NVCODEC_VERSIONS=("12.1.14.0" "12.0.16.0" "11.1.5.2" "11.0.10.2")
    
    for version in "${NVCODEC_VERSIONS[@]}"; do
        log_info "尝试版本: $version"
        
        # 清理之前的下载
        rm -rf nv-codec-headers 2>/dev/null || true
        
        # 下载特定版本
        if wget "https://gitee.com/mirrors/ffmpeg-nv-codec-headers/repository/archive/n${version}.tar.gz" -O nv-codec-headers.tar.gz 2>/dev/null; then
            tar -xzf nv-codec-headers.tar.gz
            mv nv-codec-headers-n${version} nv-codec-headers
            cd nv-codec-headers
            
            # 编译安装
            make install
            log_success "成功安装NVIDIA Video Codec SDK版本: $version"
            
            # 验证安装
            if [ -f "/usr/local/include/ffnvcodec/nvEncodeAPI.h" ]; then
                log_success "NVIDIA Video Codec SDK验证成功"
                cd /
                rm -rf "$WORKDIR"
                return 0
            fi
        fi
    done
    
    # 方法2：如果预编译版本失败，使用git克隆最新版本
    log_info "预编译版本失败，使用git克隆最新版本..."
    cd "$WORKDIR"
    rm -rf nv-codec-headers 2>/dev/null || true
    
    git clone --depth 1 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
    cd nv-codec-headers
    
    # 编译安装
    make install
    
    # 清理
    cd /
    rm -rf "$WORKDIR"
    
    log_success "NVIDIA Video Codec SDK安装完成"
    return 0
}

# 修复nvenc.c编译错误
apply_nvenc_patch() {
    log_info "应用nvenc.c编译修复..."
    
    NVENC_FILE="libavcodec/nvenc.c"
    if [ ! -f "$NVENC_FILE" ]; then
        log_error "找不到nvenc.c文件"
        return 1
    fi
    
    # 备份原始文件
    cp "$NVENC_FILE" "${NVENC_FILE}.bak"
    
    # 修复结构体成员错误（注释掉有问题的行）
    sed -i 's/hevc->pixelBitDepthMinus8 = IS_10BIT(ctx->data_pix_fmt) ? 2 : 0;/\/\/ hevc->pixelBitDepthMinus8 = IS_10BIT(ctx->data_pix_fmt) ? 2 : 0; \/\/ 注释：修复编译错误/g' "$NVENC_FILE"
    sed -i 's/av1->inputPixelBitDepthMinus8 = IS_10BIT(ctx->data_pix_fmt) ? 2 : 0;/\/\/ av1->inputPixelBitDepthMinus8 = IS_10BIT(ctx->data_pix_fmt) ? 2 : 0; \/\/ 注释：修复编译错误/g' "$NVENC_FILE"
    sed -i 's/av1->pixelBitDepthMinus8 = (IS_10BIT(ctx->data_pix_fmt) || ctx->highbitdepth) ? 2 : 0;/\/\/ av1->pixelBitDepthMinus8 = (IS_10BIT(ctx->data_pix_fmt) || ctx->highbitdepth) ? 2 : 0; \/\/ 注释：修复编译错误/g' "$NVENC_FILE"
    
    # 修复NV_ENC_BUFFER_FORMAT错误
    sed -i 's/return NV_ENC_BUFFER_FORMAT_YV12_PL;/return NV_ENC_BUFFER_FORMAT_YV12; \/\/ 修改：修复编译错误/g' "$NVENC_FILE"
    sed -i 's/return NV_ENC_BUFFER_FORMAT_NV12_PL;/return NV_ENC_BUFFER_FORMAT_NV12; \/\/ 修改：修复编译错误/g' "$NVENC_FILE"
    sed -i 's/return NV_ENC_BUFFER_FORMAT_YUV444_PL;/return NV_ENC_BUFFER_FORMAT_YUV444; \/\/ 修改：修复编译错误/g' "$NVENC_FILE"
    
    log_success "nvenc.c补丁应用完成"
}

# 4. 编译安装FFmpeg
compile_ffmpeg() {
    log_info "编译安装FFmpeg 6.1.1..."
    
    # 修复CUDA环境
    check_and_fix_cuda || log_warning "CUDA环境修复失败，将继续编译（可能不支持CUDA）"

    # 安装NVIDIA Codec SDK
    install_nvidia_codec_sdk
    
    # 清理系统FFmpeg头文件
    log_info "清理系统FFmpeg头文件..."
    rm -rf /usr/include/libavcodec \
           /usr/include/libavformat \
           /usr/include/libavutil \
           /usr/include/libswscale \
           /usr/include/libavfilter 2>/dev/null || true
    
    # 创建工作目录
    WORKDIR="/tmp/ffmpeg-build-$(date +%s)"
    mkdir -p "$WORKDIR"
    cd "$WORKDIR"
    
    # 下载源码 - 简化的进度显示
    log_info "下载FFmpeg源码(5-10min)..."
    
    # 创建工作目录
WORKDIR="/tmp/ffmpeg-build- $ (date +%s)"
mkdir -p " $ WORKDIR"
cd " $ WORKDIR"

# FFmpeg 缓存配置
FFMPEG_VERSION="6.1.1"
FFMPEG_CACHE_DIR="/opt/ffmpeg-cache"
FFMPEG_TAR="$FFMPEG_CACHE_DIR/ffmpeg-$FFMPEG_VERSION.tar.gz"

# 创建缓存目录
mkdir -p "$FFMPEG_CACHE_DIR"

# 检查是否已下载
if [ ! -f "$FFMPEG_TAR" ]; then
    log_info "FFmpeg 源码未缓存，正在下载..."
    # 安装pv用于显示进度（如果未安装）
    if ! command -v pv &> /dev/null; then
        apt-get install -y pv
    fi

    if wget --tries=3 --timeout=60 --show-progress --progress=dot:giga \
        "https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.gz" -O "$FFMPEG_TAR" 2>&1 | tail -10; then
        log_success "下载完成"
        # 验证文件
        if tar -tzf "$FFMPEG_TAR" >/dev/null 2>&1; then
            log_success "文件验证成功"
        else
            log_error "下载的文件已损坏"
            rm -f "$FFMPEG_TAR"
            return 1
        fi
    else
        log_error "FFmpeg下载失败"
        return 1
    fi
else
    log_info "使用已缓存的 FFmpeg 源码:  $FFMPEG_TAR"
fi

# 解压到工作目录
log_info "正在解压..."
tar -xzf "$FFMPEG_TAR" | pv -lep -s  $(tar -tzf "$FFMPEG_TAR" | wc -l) >/dev/null 2>&1 || \
tar -xzf "$FFMPEG_TAR"

# 进入解压后的目录
cd ffmpeg-$FFMPEG_VERSION
    
    # ... 剩余代码保持不变 ...
    # 如果是6.1.1版本，应用nvenc补丁
    if [ "$FFMPEG_VERSION" = "6.1.1" ]; then
        apply_nvenc_patch
    fi
    
    # 配置参数 - 与Dockerfile配置保持一致
    CONFIG_OPTS="--prefix=/usr/local/ffmpeg"
    CONFIG_OPTS="$CONFIG_OPTS --enable-gpl --enable-nonfree"
    CONFIG_OPTS="$CONFIG_OPTS --enable-libx264 --enable-libx265 --enable-libvpx"
    CONFIG_OPTS="$CONFIG_OPTS --enable-libmp3lame --enable-libopus --enable-libfdk-aac"
    CONFIG_OPTS="$CONFIG_OPTS --enable-libfreetype --enable-libfontconfig --enable-libass"
    CONFIG_OPTS="$CONFIG_OPTS --enable-opengl --enable-vaapi --enable-vdpau"
    CONFIG_OPTS="$CONFIG_OPTS --enable-shared --disable-static"
    CONFIG_OPTS="$CONFIG_OPTS --enable-libxml2 --enable-libdrm"
    
    # 如果CUDA可用，启用CUDA支持
    if command -v nvcc &> /dev/null || [ -d "/usr/local/cuda" ]; then
        log_info "启用CUDA支持..."
        
        # 确定CUDA路径
        CUDA_PATH="/usr/local/cuda"
        if [ -d "/usr/local/cuda-11.8" ]; then
            CUDA_PATH="/usr/local/cuda-11.8"
        fi
        
        log_info "CUDA路径: $CUDA_PATH"
        log_info "nvcc路径: $(which nvcc)"
        
        # 设置CUDA环境变量，确保当前shell能访问
        export PATH="$CUDA_PATH/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
        export CUDA_HOME="$CUDA_PATH"
        
        # 验证nvcc可用
        if command -v nvcc &> /dev/null; then
            log_info "nvcc验证通过: $(nvcc --version | head -n 1)"
            
            # 添加CUDA支持选项
            CONFIG_OPTS="$CONFIG_OPTS --enable-cuda-nvcc --enable-libnpp"
            CONFIG_OPTS="$CONFIG_OPTS --extra-cflags=-I$CUDA_PATH/include"
            CONFIG_OPTS="$CONFIG_OPTS --extra-ldflags=-L$CUDA_PATH/lib64"
            
            # 添加nvcc路径（重要！）
            CONFIG_OPTS="$CONFIG_OPTS --nvcc=$CUDA_PATH/bin/nvcc"
            
            # 对于FFmpeg 6.x，可能需要禁用一些新特性来避免编译错误
            if [ "$FFMPEG_VERSION" = "6.1.1" ] || [ "$FFMPEG_VERSION" = "6.0" ]; then
                CONFIG_OPTS="$CONFIG_OPTS --disable-encoder=nvenc_av1"  # 禁用AV1编码器，避免编译错误
            fi
        else
            log_warning "nvcc不可用，跳过CUDA支持"
        fi
    fi
    
    # 配置
    log_info "配置FFmpeg $FFMPEG_VERSION..."
    log_info "配置选项: $CONFIG_OPTS"
    echo ""
    echo "配置详情:"
    echo "----------------------------------------"
    
    # 运行配置
    ./configure $CONFIG_OPTS
    
    # 检查配置是否成功
    if [ $? -ne 0 ]; then
        log_error "FFmpeg配置失败"
        log_info "尝试不带CUDA支持重新配置..."
        
        # 移除CUDA相关配置
        CONFIG_OPTS=$(echo "$CONFIG_OPTS" | sed 's/--enable-cuda-nvcc//g')
        CONFIG_OPTS=$(echo "$CONFIG_OPTS" | sed 's/--enable-libnpp//g')
        CONFIG_OPTS=$(echo "$CONFIG_OPTS" | sed 's/--extra-cflags=-I[^ ]*//g')
        CONFIG_OPTS=$(echo "$CONFIG_OPTS" | sed 's/--extra-ldflags=-L[^ ]*//g')
        CONFIG_OPTS=$(echo "$CONFIG_OPTS" | sed 's/--nvcc=[^ ]*//g')
        CONFIG_OPTS=$(echo "$CONFIG_OPTS" | sed 's/--disable-encoder=nvenc_av1//g')
        
        log_info "重新配置选项: $CONFIG_OPTS"
        ./configure $CONFIG_OPTS
    fi
    
    if [ $? -ne 0 ]; then
        log_error "FFmpeg配置完全失败，跳过编译"
        return 1
    fi
    
    # 编译
    log_info "编译FFmpeg（使用$(nproc)个核心）..."
    make -j$(nproc)
    
    # 安装
    log_info "安装FFmpeg..."
    make install
    
    # 创建FFmpeg头文件符号链接到系统目录 - 避免头文件冲突的改进版
    log_info "设置FFmpeg头文件路径..."

    # 不再复制头文件到系统目录，避免与系统自带FFmpeg冲突
    # 改为创建pkg-config配置，让编译器通过pkg-config找到正确的头文件路径

    # 确保pkg-config能找到我们安装的FFmpeg
if [ -d "/etc/pkg-config.d" ]; then
    echo "/usr/local/ffmpeg/lib/pkgconfig" > /etc/pkg-config.d/ffmpeg.conf
else
    # Ubuntu系统通常没有/etc/pkg-config.d目录，直接设置环境变量
    echo 'export PKG_CONFIG_PATH=/usr/local/ffmpeg/lib/pkgconfig: $ PKG_CONFIG_PATH' >> /etc/profile.d/media_env.sh
fi

    log_info "FFmpeg头文件设置完成（通过pkg-config管理，避免冲突）"
    
    # 创建库符号链接（与Dockerfile完全一致）
    log_info "创建FFmpeg库符号链接..."
    ln -sf /usr/local/ffmpeg/lib/libavdevice.so* /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
    ln -sf /usr/local/ffmpeg/lib/libavformat.so* /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
    ln -sf /usr/local/ffmpeg/lib/libavcodec.so* /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
    ln -sf /usr/local/ffmpeg/lib/libavutil.so* /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
    ln -sf /usr/local/ffmpeg/lib/libswscale.so* /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
    ln -sf /usr/local/ffmpeg/lib/libswresample.so* /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
    
    # 创建命令软链接，确保可以直接使用ffmpeg命令
    log_info "创建FFmpeg命令软链接..."
    ln -sf /usr/local/ffmpeg/bin/ffmpeg /usr/local/bin/ffmpeg
    ln -sf /usr/local/ffmpeg/bin/ffprobe /usr/local/bin/ffprobe
    ln -sf /usr/local/ffmpeg/bin/ffplay /usr/local/bin/ffplay 2>/dev/null || true
    
    # 设置库路径和pkg-config（与Dockerfile完全一致）
    echo "/usr/local/ffmpeg/lib" > /etc/ld.so.conf.d/ffmpeg.conf
    ldconfig
    
    # 验证安装
    log_info "验证FFmpeg安装..."
    if [ -x "/usr/local/ffmpeg/bin/ffmpeg" ]; then
        /usr/local/ffmpeg/bin/ffmpeg -version | head -n 2
        log_success "FFmpeg安装成功"
    else
        log_error "FFmpeg安装失败"
        return 1
    fi
    
    # 清理
    cd /
    rm -rf "$WORKDIR"
    
        # === 新增：创建符号链接让CMake能找到FFmpeg头文件 ===
    log_info "创建FFmpeg头文件符号链接到 /usr/local/include..."
    ln -sf /usr/local/ffmpeg/include/libavutil /usr/local/include/libavutil 2>/dev/null || true
    ln -sf /usr/local/ffmpeg/include/libavcodec /usr/local/include/libavcodec 2>/dev/null || true
    ln -sf /usr/local/ffmpeg/include/libavformat /usr/local/include/libavformat 2>/dev/null || true
    ln -sf /usr/local/ffmpeg/include/libswscale /usr/local/include/libswscale 2>/dev/null || true
    ln -sf /usr/local/ffmpeg/include/libswresample /usr/local/include/libswresample 2>/dev/null || true
    ln -sf /usr/local/ffmpeg/include/libavfilter /usr/local/include/libavfilter 2>/dev/null || true
    ln -sf /usr/local/ffmpeg/include/libavdevice /usr/local/include/libavdevice 2>/dev/null || true
    # ========================================================

    log_success "FFmpeg编译安装完成"
}

# 5. 安装Python包 - 只安装必要的，与Dockerfile保持一致
install_python_packages() {
    log_info "安装Python包..."
    
    # 升级pip
    python3 -m pip install --upgrade pip setuptools wheel
    
    # 只安装Dockerfile中提到的包，移除torch等不需要的包
    python3 -m pip install \
        numpy \
        opencv-python
    
    log_success "Python包安装完成"
}

# 6. 设置环境变量
setup_environment() {
    log_info "设置环境变量..."
    # 创建环境变量文件（与 Dockerfile 保持一致）
    cat > /etc/profile.d/media_env.sh << 'EOF'
#!/bin/sh
# 多媒体处理环境变量
export FFMPEG_HOME=/usr/local/ffmpeg
export PATH=$FFMPEG_HOME/bin:$PATH
export LD_LIBRARY_PATH=$FFMPEG_HOME/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=/usr/local/ffmpeg/lib/pkgconfig:$PKG_CONFIG_PATH
# CUDA 环境变量（仅设置必要路径）
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
fi
EOF

    # 立即生效
    chmod +x /etc/profile.d/media_env.sh
    source /etc/profile.d/media_env.sh

    # 直接更新当前 shell 的环境变量
    export PATH=/usr/local/ffmpeg/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/ffmpeg/lib:$LD_LIBRARY_PATH
    export PKG_CONFIG_PATH=/usr/local/ffmpeg/lib/pkgconfig:$PKG_CONFIG_PATH

    # 添加到用户配置文件
    if [ -f "/root/.bashrc" ]; then
        echo 'source /etc/profile.d/media_env.sh' >> /root/.bashrc
    fi
    if [ -f "/home/$SUDO_USER/.bashrc" ] && [ ! -z "$SUDO_USER" ]; then
        echo 'source /etc/profile.d/media_env.sh' >> "/home/$SUDO_USER/.bashrc"
        chown $SUDO_USER:$SUDO_USER "/home/$SUDO_USER/.bashrc"
    fi
    log_success "环境变量设置完成（已移除 CPLUS_INCLUDE_PATH/CPATH 污染）"
}

# 7. 验证安装 - 与Dockerfile验证一致
verify_installation() {
    echo ""
    log_info "=== 验证安装 ==="
    
    echo "1. 系统信息:"
    echo "   Ubuntu: $VERSION"
    echo "   内核: $(uname -r)"
    
    echo ""
    echo "2. CUDA检查:"
    if command -v nvcc &> /dev/null; then
        nvcc --version | head -n 4
    else
        echo "   未安装或未检测到CUDA"
    fi
    
    echo ""
    echo "3. FFmpeg检查:"
    if command -v ffmpeg &> /dev/null; then
        ffmpeg -version | head -n 3
        echo ""
        echo "   检查libavutil/frame.h文件位置:"
        find /usr -name "frame.h" 2>/dev/null | grep libavutil | sort | head -5
    else
        echo "   FFmpeg安装失败或未在PATH中"
        echo "   尝试使用绝对路径: /usr/local/ffmpeg/bin/ffmpeg"
        if [ -x "/usr/local/ffmpeg/bin/ffmpeg" ]; then
            /usr/local/ffmpeg/bin/ffmpeg -version | head -n 2
        fi
    fi
    
    echo ""
    echo "4. Qt检查:"
    if command -v qmake &> /dev/null; then
        qmake --version | head -n 2
    else
        echo "   qmake未找到"
    fi
    
    echo ""
    echo "5. OpenCV检查:"
    python3 -c "import cv2; print(f'   OpenCV版本: {cv2.__version__}')" 2>/dev/null || \
    echo "   OpenCV Python绑定未安装"
    
    echo ""
    echo "6. 关键依赖检查:"
    echo "   libspdlog: $(ldconfig -p | grep -c libspdlog) 个库"
    echo "   libnpp: $(ldconfig -p | grep -c libnpp) 个库"
    echo "   libavcodec: $(ldconfig -p | grep -c libavcodec) 个库"
    echo "   libavutil: $(ldconfig -p | grep -c libavutil) 个库"
    
    echo ""
    log_success "验证完成"
}

# 8. 清理工作
cleanup() {
    log_info "清理临时文件..."
    
    apt-get autoremove -y
    apt-get clean
    rm -rf /var/lib/apt/lists/*
    rm -rf /tmp/*
    
    log_success "清理完成"
}

# 9. 创建测试脚本
create_test_script() {
    cat > /usr/local/bin/test_media_env.sh << 'EOF'
#!/bin/bash

echo "=== 多媒体环境测试 ==="
echo ""

# 测试CUDA
echo "1. 测试CUDA:"
if command -v nvcc &> /dev/null; then
    echo "   CUDA可用"
    nvcc --version | head -n 1
else
    echo "   CUDA不可用"
fi

echo ""

# 测试FFmpeg
echo "2. 测试FFmpeg:"
if command -v ffmpeg &> /dev/null; then
    echo "   FFmpeg可用"
    ffmpeg -version | head -n 1
    echo ""
    echo "   支持的硬件加速:"
    ffmpeg -hwaccels 2>/dev/null | tail -n +2 2>/dev/null || echo "   无法获取硬件加速信息"
else
    echo "   FFmpeg不可用"
    echo "   尝试使用绝对路径..."
    if [ -x "/usr/local/ffmpeg/bin/ffmpeg" ]; then
        echo "   FFmpeg在 /usr/local/ffmpeg/bin/ffmpeg 可用"
        /usr/local/ffmpeg/bin/ffmpeg -version | head -n 1
    fi
fi

echo ""

# 测试Qt
echo "3. 测试Qt:"
if command -v qmake &> /dev/null; then
    echo "   Qt可用"
    qmake --version | head -n 1
else
    echo "   Qt不可用"
fi

echo ""

# 测试OpenCV
echo "4. 测试OpenCV:"
python3 -c "
try:
    import cv2
    print('   OpenCV可用')
    print(f'   版本: {cv2.__version__}')
except ImportError:
    print('   OpenCV不可用')
"

echo ""
echo "=== 测试完成 ==="
EOF
    
    chmod +x /usr/local/bin/test_media_env.sh
    log_success "测试脚本创建完成: test_media_env.sh"
}

# 主安装流程
main() {
    echo ""
    log_info "开始安装流程..."
    echo ""
    
    START_TIME=$(date +%s)
    
    # 执行安装步骤
    update_system
    install_cuda
    install_all_dependencies
    compile_ffmpeg
    install_python_packages
    setup_environment
    create_test_script
    cleanup
    verify_installation
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    log_success "=== 安装完成 ==="
    echo ""
    echo "安装总结:"
    echo "  总耗时: $((DURATION / 60))分钟 $((DURATION % 60))秒"
    echo ""
    echo "重要提示:"
    echo "  1. FFmpeg已安装到: /usr/local/ffmpeg/bin/ffmpeg"
    echo "  2. FFmpeg头文件已复制到: /usr/include/libavutil/"
    echo "  3. 已创建软链接到: /usr/local/bin/ffmpeg"
    echo "  4. 环境变量已设置，请重新启动终端或运行以下命令:"
    echo "     source /etc/profile.d/media_env.sh"
    echo "  5. 运行测试: test_media_env.sh"
    echo "  6. 验证命令: ffmpeg -version, nvcc --version"
    echo "  7. 验证头文件: ls /usr/include/libavutil/frame.h"
    echo ""
}

# 错误处理
handle_error() {
    log_error "安装过程中出现错误!"
    log_error "错误发生在: $1"
    log_error "退出状态: $2"
    
    echo ""
    echo "故障排除建议:"
    echo "  1. 检查网络连接"
    echo "  2. 确保有足够的磁盘空间"
    echo "  3. 查看详细日志: tail -f /var/log/install.log"
    
    exit 1
}

# 设置trap捕获错误
trap 'handle_error "${BASH_SOURCE[0]}:${LINENO}" "$?"' ERR

# 创建日志文件
exec > >(tee -a /var/log/media_env_install.log) 2>&1

# 运行主函数
main
