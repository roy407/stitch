#!/bin/bash

# 默认参数
CAM_JSON=""
ENABLE_KERNEL_TEST=0
BUILD_DIR="build"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            CAM_JSON="$2"
            shift 2
            ;;
        -k|--kernel_test)
            ENABLE_KERNEL_TEST=1
            shift
            ;;
        -h|--help)
            echo "用法: ./start_split.sh [选项]"
            echo ""
            echo "选项:"
            echo "  -c, --config <名称>     指定配置 (例如: cam10, cam5, cam10_jetson)"
            echo "  -k, --kernel_test      启用 kernel 测试"
            echo "  -h, --help             显示帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 如果没有提供 config，通过提示让用户输入或者设置默认值
if [ -z "$CAM_JSON" ]; then
    echo "未指定配置，默认使用 cam10"
    CAM_JSON="cam10"
fi

# 根据简写名称映射到具体的 JSON 文件路径
case "$CAM_JSON" in
    "cam2")
        CONFIG_FILE="resource/cam2.json"
        ;;
    "hk5")
        CONFIG_FILE="resource/hk5.json"
        ;;
    "cam5")
        CONFIG_FILE="resource/cam5.json"
        ;;
    "cam10")
        CONFIG_FILE="resource/cam10.json"
        ;;
    "cam10_jetson")
        CONFIG_FILE="resource/cam10_jetson.json"
        ;;
    "cam_test")
        CONFIG_FILE="resource/cam_test.json"
        ;;
    *".json")
        # 如果用户直接传了 xxx.json
        CONFIG_FILE="$CAM_JSON"
        ;;
    *)
        echo "错误: 不支持的相机配置名称: $CAM_JSON"
        exit 1
        ;;
esac

echo "使用配置文件: $CONFIG_FILE"

# 1. 编译项目
echo "正在编译..."
mkdir -p $BUILD_DIR
cd $BUILD_DIR
# 注意：这里我们 build 整个项目，会生成 stitch_cam 和 stitch_ui
cmake .. -DENABLE_KERNEL_TEST=$ENABLE_KERNEL_TEST || { echo "CMake 失败"; exit 1; }
make -j$(nproc) || { echo "编译失败"; exit 1; }

# 定义清理函数，确保脚本退出时清理后台进程
cleanup() {
    echo ""
    echo "正在关闭后台进程..."
    if [ -n "$PID_CAM" ]; then
        kill $PID_CAM 2>/dev/null
        wait $PID_CAM 2>/dev/null
        echo "stitch_cam (PID $PID_CAM) 已停止"
    fi
}
# 捕获 EXIT, INT(Ctrl+C), TERM 信号
trap cleanup EXIT INT TERM

# 2. 启动后台处理进程 (stitch_cam)
echo "----------------------------------------"
echo "启动后台处理进程: stitch_cam"
./stitch_cam "$CONFIG_FILE" &
PID_CAM=$!

# 等待几秒钟让后台初始化 (主要是共享内存创建)
echo "等待后台初始化 (3秒)..."
sleep 3

# 检查后台进程是否还活着
if ! kill -0 $PID_CAM 2>/dev/null; then
    echo "错误: stitch_cam 启动失败或意外退出"
    exit 1
fi

# 3. 启动前台 UI 进程 (stitch_ui)
echo "----------------------------------------"
echo "启动前台显示进程: stitch_ui"
./stitch_ui "$CONFIG_FILE"

# UI 退出后，脚本继续执行到 cleanup，自动杀掉 cam 进程
echo "----------------------------------------"
echo "UI 进程已退出"
