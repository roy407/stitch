#!/bin/bash

# stitch_build.sh - 智能构建脚本

set -e  # 出错时退出

# 配置
BUILD_DIR="build"
JOBS=$(nproc)
CONFIG_FILE=""
BUILD_SHARED_LIB=0
ENABLE_KERNEL_TEST=0
CLEAN_BUILD=0

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 解析命令行参数
show_help() {
    echo "智能构建脚本 - 快速构建 C 或 C++ 版本"
    echo ""
    echo "用法: ./stitch_build.sh [选项]"
    echo ""
    echo "选项:"
    echo "  -c, --config <name>     指定相机配置 (cam2, cam5, cam10 等)"
    echo "  --c-version             构建 C 版本 (默认构建 C++ 版本)"
    echo "  -k, --kernel-test       启用 kernel 测试"
    echo "  -j, --jobs <num>        指定并行构建任务数 (默认: CPU核心数)"
    echo "  --clean                 清理构建目录后重新构建"
    echo "  --build-only            仅构建，不运行程序"
    echo "  --configure-only        仅配置，不构建"
    echo "  -h, --help              显示帮助信息"
    echo ""
    echo "示例:"
    echo "  ./stitch_build.sh -c cam5                    # 构建并运行C++版本"
    echo "  ./stitch_build.sh -c cam5 --c-version        # 构建并运行C版本"
    echo "  ./stitch_build.sh -c cam5 --clean -j 8       # 清理后并行构建"
    echo "  ./stitch_build.sh --c-version --build-only   # 仅构建C版本"
    exit 0
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --c-version)
            BUILD_SHARED_LIB=1
            shift
            ;;
        -k|--kernel-test)
            ENABLE_KERNEL_TEST=1
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        --build-only)
            BUILD_ONLY=1
            shift
            ;;
        --configure-only)
            CONFIGURE_ONLY=1
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            error "未知参数: $1"
            exit 1
            ;;
    esac
done

# 验证配置
if [[ -z "$CONFIG_FILE" ]] && [[ -z "$BUILD_ONLY" ]] && [[ -z "$CONFIGURE_ONLY" ]]; then
    error "必须指定相机配置（使用 -c 参数）"
    echo "可用的配置: cam2, hk5, cam5, cam10, cam10_jetson, cam_test"
    exit 1
fi

# 映射配置文件
case "$CONFIG_FILE" in
    cam2) CONFIG_PATH="resource/cam2.json" ;;
    hk5) CONFIG_PATH="resource/hk5.json" ;;
    cam5) CONFIG_PATH="resource/cam5.json" ;;
    cam10) CONFIG_PATH="resource/cam10.json" ;;
    cam10_jetson) CONFIG_PATH="resource/cam10_jetson.json" ;;
    cam_test) CONFIG_PATH="resource/cam_test.json" ;;
    "")
        if [[ -z "$BUILD_ONLY" ]] && [[ -z "$CONFIGURE_ONLY" ]]; then
            error "无效的配置: $CONFIG_FILE"
            exit 1
        fi
        ;;
    *)
        error "不支持的配置: $CONFIG_FILE"
        exit 1
        ;;
esac

# 清理构建目录
if [[ $CLEAN_BUILD -eq 1 ]]; then
    info "清理构建目录: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

# 创建构建目录
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 检查是否需要重新配置
NEED_RECONFIGURE=1
if [[ -f "CMakeCache.txt" ]]; then
    # 检查当前配置是否与上次相同
    OLD_C_VERSION=$(grep "BUILD_SHARED_LIB" CMakeCache.txt | cut -d= -f2)
    OLD_KERNEL_TEST=$(grep "ENABLE_KERNEL_TEST" CMakeCache.txt | cut -d= -f2)
    
    if [[ "$OLD_C_VERSION" == "$BUILD_SHARED_LIB" ]] && [[ "$OLD_KERNEL_TEST" == "$ENABLE_KERNEL_TEST" ]]; then
        info "使用现有CMake配置"
        NEED_RECONFIGURE=0
    else
        info "配置已更改，重新配置CMake"
    fi
fi

# 运行CMake配置
if [[ $NEED_RECONFIGURE -eq 1 ]]; then
    info "运行CMake配置..."
    cmake .. \
        -DENABLE_KERNEL_TEST=$ENABLE_KERNEL_TEST \
        -DBUILD_SHARED_LIB=$BUILD_SHARED_LIB
    
    if [[ $? -ne 0 ]]; then
        error "CMake配置失败"
        exit 1
    fi
fi

# 仅配置模式
if [[ -n "$CONFIGURE_ONLY" ]]; then
    success "CMake配置完成"
    exit 0
fi

# 构建
info "开始构建（使用 $JOBS 个并行任务）..."
start_time=$(date +%s)
make -j$JOBS
end_time=$(date +%s)
build_time=$((end_time - start_time))

success "构建完成！耗时: ${build_time}秒"

# 检查构建结果
EXECUTABLE="./bin/stitch_app"
if [[ ! -f "$EXECUTABLE" ]]; then
    EXECUTABLE="./stitch_app"
fi

if [[ ! -f "$EXECUTABLE" ]]; then
    error "未找到可执行文件"
    exit 1
fi

# 显示文件信息
info "生成的可执行文件:"
echo "  路径: $(realpath "$EXECUTABLE")"
echo "  大小: $(du -h "$EXECUTABLE" | cut -f1)"
echo "  类型: $(file -b "$EXECUTABLE")"

# 仅构建模式
if [[ -n "$BUILD_ONLY" ]]; then
    success "构建完成"
    exit 0
fi

# 运行程序
if [[ -n "$CONFIG_PATH" ]]; then
    info "运行程序..."
    info "配置文件: $CONFIG_PATH"
    
    # 检查配置文件是否存在
    if [[ ! -f "../$CONFIG_PATH" ]]; then
        warning "配置文件不存在: $CONFIG_PATH"
        info "尝试在构建目录中查找..."
        if [[ -f "$CONFIG_PATH" ]]; then
            info "使用构建目录中的配置文件"
        else
            error "找不到配置文件"
            exit 1
        fi
    fi
    
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}       启动 stitch 程序${NC}"
    echo -e "${GREEN}========================================${NC}\n"
    
    # 运行程序
    "$EXECUTABLE" "$CONFIG_PATH"
    
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}       程序运行结束${NC}"
    echo -e "${GREEN}========================================${NC}\n"
    
    # 检查是否有绘图脚本
    if [[ -f "../scripts/plot_timing.py" ]]; then
        info "运行绘图脚本..."
        python3 ../scripts/plot_timing.py
    fi
fi

success "所有任务完成"