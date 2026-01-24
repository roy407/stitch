#!/bin/bash
# ========================================
# 环境配置脚本 - 用于新安装的 Ubuntu 系统
# 自动安装项目所需的所有依赖
# ========================================

set -e  # 遇到错误立即退出

# 颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() { echo -e "${BLUE}[ℹ] $1${NC}"; }
print_success() { echo -e "${GREEN}[✓] $1${NC}"; }
print_error() { echo -e "${RED}[✗] $1${NC}"; }
print_warning() { echo -e "${YELLOW}[!] $1${NC}"; }

# 检查命令是否存在
check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# 检查是否为 Ubuntu 系统
check_ubuntu() {
    if [ ! -f /etc/os-release ]; then
        print_error "无法检测系统类型，请确保在 Ubuntu 系统上运行此脚本"
        exit 1
    fi
    
    source /etc/os-release
    if [[ "$ID" != "ubuntu" ]]; then
        print_warning "检测到系统为 $ID，此脚本主要针对 Ubuntu 系统"
        read -p "是否继续？(y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "检测到 Ubuntu 系统: $VERSION"
    fi
}

# 更新系统包
update_system() {
    print_info "更新系统包列表..."
    sudo apt update
    print_success "系统包列表更新完成"
}

# 安装基础构建工具
install_build_tools() {
    print_info "安装基础构建工具..."
    
    local packages=(
        "build-essential"
        "cmake"
        "git"
        "pkg-config"
        "libtool"
        "autoconf"
        "automake"
        "yasm"
        "nasm"
        "wget"
        "curl"
        "python3"
        "python3-pip"
    )
    
    for pkg in "${packages[@]}"; do
        if ! dpkg -l | grep -q "^ii  $pkg "; then
            print_info "安装 $pkg..."
            sudo apt install -y "$pkg"
        else
            print_success "$pkg 已安装"
        fi
    done
    
    print_success "基础构建工具安装完成"
}

# 检查并安装 NVIDIA 驱动
check_nvidia_driver() {
    print_info "检查 NVIDIA 驱动..."
    
    if check_command nvidia-smi; then
        local driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
        print_success "NVIDIA 驱动已安装，版本: $driver_version"
        
        # 检查驱动版本是否满足要求（≥ 535）
        local major_version=$(echo "$driver_version" | cut -d. -f1)
        if [ "$major_version" -lt 535 ]; then
            print_warning "驱动版本 $driver_version 可能不满足要求（建议 ≥ 535）"
            print_warning "如需更新驱动，请运行: sudo ubuntu-drivers autoinstall"
        fi
    else
        print_warning "未检测到 NVIDIA 驱动"
        print_info "正在安装 NVIDIA 驱动..."
        sudo ubuntu-drivers autoinstall
        print_warning "驱动安装后需要重启系统，请重启后再次运行此脚本"
        exit 1
    fi
}

# 检查并安装 CUDA
check_cuda() {
    print_info "检查 CUDA 安装..."
    
    if check_command nvcc; then
        local cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        print_success "CUDA 已安装，版本: $cuda_version"
        
        # 检查 CUDA 版本是否满足要求（≥ 11.8）
        local major=$(echo "$cuda_version" | cut -d. -f1)
        local minor=$(echo "$cuda_version" | cut -d. -f2)
        if [ "$major" -lt 11 ] || ([ "$major" -eq 11 ] && [ "$minor" -lt 8 ]); then
            print_warning "CUDA 版本 $cuda_version 不满足要求（需要 ≥ 11.8）"
            print_info "请从 NVIDIA 官网下载并安装 CUDA 11.8 或更高版本"
            print_info "下载地址: https://developer.nvidia.com/cuda-downloads"
        fi
    else
        print_warning "未检测到 CUDA"
        print_info "请从 NVIDIA 官网下载并安装 CUDA 11.8 或更高版本"
        print_info "下载地址: https://developer.nvidia.com/cuda-downloads"
        print_info "安装完成后，请确保 CUDA 路径已添加到环境变量中"
        
        # 尝试设置常见的 CUDA 路径
        if [ -d "/usr/local/cuda" ]; then
            print_info "检测到 /usr/local/cuda 目录，尝试添加到环境变量..."
            export PATH=/usr/local/cuda/bin:$PATH
            export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
        fi
    fi
    
    # 检查 CUDA 环境变量
    if [ -z "$CUDA_HOME" ] && [ -d "/usr/local/cuda" ]; then
        print_info "设置 CUDA 环境变量..."
        echo "" >> ~/.bashrc
        echo "# CUDA Environment" >> ~/.bashrc
        echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
        echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
        print_success "CUDA 环境变量已添加到 ~/.bashrc，请运行 source ~/.bashrc 或重新打开终端"
    fi
}

# 安装 Qt5
install_qt5() {
    print_info "检查 Qt5 安装..."
    
    if check_command qmake; then
        local qt_version=$(qmake -v | grep "Using Qt version" | sed 's/.*Using Qt version \([0-9]\+\.[0-9]\+\).*/\1/')
        print_success "Qt5 已安装，版本: $qt_version"
    else
        print_info "安装 Qt5..."
        sudo apt install -y \
            qtbase5-dev \
            qtbase5-dev-tools \
            qttools5-dev \
            qttools5-dev-tools \
            qt5-qmake \
            libqt5widgets5 \
            libqt5gui5 \
            libqt5core5a
        
        print_success "Qt5 安装完成"
    fi
}

# 安装 OpenGL 开发库
install_opengl() {
    print_info "安装 OpenGL 开发库..."
    
    sudo apt install -y \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        freeglut3-dev \
        libglew-dev
    
    print_success "OpenGL 开发库安装完成"
}

# 安装 spdlog
install_spdlog() {
    print_info "检查 spdlog 安装..."
    
    if pkg-config --exists spdlog 2>/dev/null; then
        print_success "spdlog 已安装"
    else
        print_info "安装 spdlog..."
        sudo apt install -y libspdlog-dev
        print_success "spdlog 安装完成"
    fi
}

# 安装 nlohmann/json
install_nlohmann_json() {
    print_info "检查 nlohmann/json 安装..."
    
    if pkg-config --exists nlohmann_json 2>/dev/null || [ -f "/usr/include/nlohmann/json.hpp" ]; then
        print_success "nlohmann/json 已安装"
    else
        print_info "安装 nlohmann/json..."
        sudo apt install -y nlohmann-json3-dev
        print_success "nlohmann/json 安装完成"
    fi
}

# 检查 FFmpeg
check_ffmpeg() {
    print_info "检查 FFmpeg 安装..."
    
    if check_command ffmpeg; then
        local ffmpeg_version=$(ffmpeg -version | head -n1 | sed 's/.*ffmpeg version \([^ ]\+\).*/\1/')
        print_success "FFmpeg 已安装，版本: $ffmpeg_version"
        
        # 检查是否支持硬件编解码
        print_info "检查 FFmpeg 硬件编解码支持..."
        if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q "h264_nvenc"; then
            print_success "FFmpeg 支持 NVIDIA 硬件编码 (h264_nvenc)"
        else
            print_warning "FFmpeg 可能不支持 NVIDIA 硬件编码"
        fi
        
        if ffmpeg -hide_banner -decoders 2>/dev/null | grep -q "h264_cuvid"; then
            print_success "FFmpeg 支持 NVIDIA 硬件解码 (h264_cuvid)"
        else
            print_warning "FFmpeg 可能不支持 NVIDIA 硬件解码"
        fi
        
        # 检查头文件位置
        if [ -f "/usr/local/include/libavcodec/avcodec.h" ]; then
            print_success "FFmpeg 头文件位于 /usr/local/include（符合项目要求）"
        elif [ -f "/usr/include/libavcodec/avcodec.h" ]; then
            print_warning "FFmpeg 头文件位于 /usr/include，但项目期望在 /usr/local/include"
            print_warning "如果编译时出现找不到头文件的错误，可能需要："
            print_warning "1. 重新编译 FFmpeg 并安装到 /usr/local"
            print_warning "2. 或修改 CMakeLists.txt 中的 FFMPEG_INCLUDE_DIRS"
        fi
    else
        print_error "未检测到 FFmpeg"
        print_warning "FFmpeg 需要手动编译以支持硬件编解码"
        print_info "编译 FFmpeg 的步骤："
        echo ""
        echo "1. 安装依赖："
        echo "   sudo apt install -y \\"
        echo "     libx264-dev libx265-dev libvpx-dev libfdk-aac-dev \\"
        echo "     libmp3lame-dev libopus-dev libvorbis-dev \\"
        echo "     libass-dev libfreetype6-dev libsdl2-dev \\"
        echo "     libva-dev libvdpau-dev"
        echo ""
        echo "2. 下载 FFmpeg 源码："
        echo "   git clone https://git.ffmpeg.org/ffmpeg.git"
        echo "   cd ffmpeg"
        echo ""
        echo "3. 配置并编译（支持 NVIDIA 硬件编解码）："
        echo "   ./configure \\"
        echo "     --enable-nonfree \\"
        echo "     --enable-gpl \\"
        echo "     --enable-libx264 \\"
        echo "     --enable-libx265 \\"
        echo "     --enable-libvpx \\"
        echo "     --enable-libfdk-aac \\"
        echo "     --enable-libmp3lame \\"
        echo "     --enable-libopus \\"
        echo "     --enable-libvorbis \\"
        echo "     --enable-libass \\"
        echo "     --enable-libfreetype \\"
        echo "     --enable-libsdl2 \\"
        echo "     --enable-cuda \\"
        echo "     --enable-cuvid \\"
        echo "     --enable-nvenc \\"
        echo "     --enable-libnpp \\"
        echo "     --extra-cflags=-I/usr/local/cuda/include \\"
        echo "     --extra-ldflags=-L/usr/local/cuda/lib64 \\"
        echo "     --prefix=/usr/local"
        echo ""
        echo "   make -j\$(nproc)"
        echo "   sudo make install"
        echo ""
        echo "4. 更新库路径："
        echo "   sudo ldconfig"
        echo ""
        read -p "是否现在安装 FFmpeg 编译依赖？(y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo apt install -y \
                libx264-dev libx265-dev libvpx-dev libfdk-aac-dev \
                libmp3lame-dev libopus-dev libvorbis-dev \
                libass-dev libfreetype6-dev libsdl2-dev \
                libva-dev libvdpau-dev
            print_success "FFmpeg 编译依赖已安装"
        fi
    fi
}

# 安装 Git hooks
install_git_hooks() {
    print_info "安装 Git hooks..."
    
    if [ ! -d ".git" ]; then
        print_warning "当前目录不是 Git 仓库，跳过 Git hooks 安装"
        return
    fi
    
    if [ ! -f "./scripts/git-hooks/install-git-hooks.sh" ]; then
        print_warning "未找到 install-git-hooks.sh，跳过 Git hooks 安装"
        return
    fi
    
    chmod +x ./scripts/git-hooks/install-git-hooks.sh
    ./scripts/git-hooks/install-git-hooks.sh
    
    if [ -f ".git/hooks/commit-msg" ]; then
        print_success "Git hooks 安装完成"
    else
        print_warning "Git hooks 安装可能失败，请手动检查"
    fi
}

# 验证环境
verify_environment() {
    print_info "验证环境配置..."
    
    local errors=0
    
    # 检查关键命令
    local commands=("cmake" "make" "git" "qmake" "nvcc")
    for cmd in "${commands[@]}"; do
        if check_command "$cmd"; then
            print_success "$cmd 可用"
        else
            print_error "$cmd 不可用"
            ((errors++))
        fi
    done
    
    # 检查 CUDA
    if check_command nvidia-smi; then
        print_success "NVIDIA 驱动可用"
    else
        print_error "NVIDIA 驱动不可用"
        ((errors++))
    fi
    
    # 检查 Qt5
    if pkg-config --exists Qt5Widgets 2>/dev/null; then
        print_success "Qt5 Widgets 可用"
    else
        print_warning "Qt5 Widgets 可能不可用（如果已安装 Qt5，可能是 pkg-config 配置问题）"
    fi
    
    # 检查 FFmpeg
    if check_command ffmpeg; then
        print_success "FFmpeg 可用"
    else
        print_error "FFmpeg 不可用（需要手动编译）"
        ((errors++))
    fi
    
    if [ $errors -eq 0 ]; then
        print_success "环境验证完成，所有关键组件已就绪"
        return 0
    else
        print_error "环境验证失败，有 $errors 个问题需要解决"
        return 1
    fi
}

# 主函数
main() {
    echo "========================================"
    echo "   Stitch 项目环境配置脚本"
    echo "========================================"
    echo ""
    
    # 检查是否为 root 用户
    if [ "$EUID" -eq 0 ]; then
        print_error "请不要使用 root 用户运行此脚本"
        exit 1
    fi
    
    # 检查系统
    check_ubuntu
    
    # 更新系统
    update_system
    
    # 安装基础工具
    install_build_tools
    
    # 检查 NVIDIA 驱动
    check_nvidia_driver
    
    # 检查 CUDA
    check_cuda
    
    # 安装 Qt5
    install_qt5
    
    # 安装 OpenGL
    install_opengl
    
    # 安装 spdlog
    install_spdlog
    
    # 安装 nlohmann/json
    install_nlohmann_json
    
    # 检查 FFmpeg
    check_ffmpeg
    
    # 安装 Git hooks
    install_git_hooks
    
    # 验证环境
    echo ""
    echo "========================================"
    verify_environment
    echo "========================================"
    echo ""
    
    print_info "环境配置完成！"
    print_info "如果修改了环境变量（如 CUDA），请运行: source ~/.bashrc"
    print_info "或者重新打开终端窗口"
    echo ""
    print_info "下一步："
    print_info "1. 如果 FFmpeg 未安装，请按照提示手动编译 FFmpeg"
    print_info "2. 运行编译脚本: bash build_install.sh"
    print_info "3. 或直接运行: bash start_camera.sh -c <配置文件名>"
    echo ""
}

# 运行主函数
main