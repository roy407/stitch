#!/bin/bash
# ========================================
# 简化版 C++ 项目构建脚本
# 支持基本构建、测试和安装功能
# ========================================

set -e  # 出错退出

# 颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_green() { echo -e "${GREEN}[✓] $1${NC}"; }
print_red() { echo -e "${RED}[✗] $1${NC}"; }
print_yellow() { echo -e "${YELLOW}[!] $1${NC}"; }

# 默认配置
BUILD_TYPE="Release"
BUILD_DIR="build"
INSTALL_DIR="install"
ENABLE_TESTS="OFF"
BUILD_SHARED_LIB=1
CLEAN=false
JOBS=$(nproc)  # 使用CPU核心数

# 显示帮助
show_help() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -h, --help      显示帮助信息"
    echo "  -c, --clean     清理构建目录"
    echo "  -d, --debug     使用Debug模式构建"
    echo "  -t, --test      构建并运行测试"
    echo "  -j N            使用N个并行任务 (默认: CPU核心数)"
    echo "  --prefix DIR    指定安装目录 (默认: ./install)"
    echo "  --build-dir DIR 指定构建目录 (默认: ./build)"
    echo ""
    echo "示例:"
    echo "  $0               # Release构建"
    echo "  $0 -d            # Debug构建"
    echo "  $0 -d -t         # Debug构建并运行测试"
    echo "  $0 -c            # 清理构建目录"
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -t|--test)
            ENABLE_TESTS="ON"
            shift
            ;;
        -j)
            JOBS="$2"
            shift 2
            ;;
        --prefix)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        *)
            print_red "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 清理构建目录
if [ "$CLEAN" = true ]; then
    print_yellow "清理构建目录: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
    print_green "清理完成"
    exit 0
fi

# 检查必要工具
check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_red "需要 $1 但未找到，请先安装"
        exit 1
    fi
}

check_command cmake
check_command make

# 显示配置信息
echo "========================================"
echo "构建配置:"
echo "  构建类型: $BUILD_TYPE"
echo "  构建目录: $BUILD_DIR"
echo "  安装目录: $INSTALL_DIR"
echo "  启用测试: $ENABLE_TESTS"
echo "  并行任务: $JOBS"
echo "========================================"

# 创建构建目录
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 配置项目
print_green "配置CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="../$INSTALL_DIR" \
    -DBUILD_TESTING="$ENABLE_TESTS" \
    -DBUILD_SHARED_LIB=$BUILD_SHARED_LIB

# 构建项目
print_green "构建项目..."
make -j"$JOBS"

# 运行测试（如果启用）
if [ "$ENABLE_TESTS" = "ON" ]; then
    print_green "运行测试..."
    ctest --output-on-failure
    if [ $? -eq 0 ]; then
        print_green "所有测试通过"
    else
        print_red "测试失败"
        exit 1
    fi
fi

# 安装项目
print_green "安装到: ../$INSTALL_DIR"
make install

cd ..

# 显示安装结果
echo "========================================"
print_green "构建完成!"
echo ""
echo "安装内容:"
find "$INSTALL_DIR" -type f | head -10
if [ $(find "$INSTALL_DIR" -type f | wc -l) -gt 10 ]; then
    echo "... 还有更多文件"
fi
echo "========================================"