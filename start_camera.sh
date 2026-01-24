#!/bin/bash

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
            echo "用法: ./start_camera.sh [选项]"
            echo ""
            echo "选项:"
            echo "  -c, --config <文件>     指定相机配置文件"
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

# 根据名字选择配置文件
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
    *)
        echo "错误: 不支持的相机格式: $CAM_JSON"
        exit 1
        ;;
esac

mkdir -p $BUILD_DIR
cd $BUILD_DIR && cmake .. -DENABLE_KERNEL_TEST=$ENABLE_KERNEL_TEST && make

echo "使用配置文件: $CONFIG_FILE"
echo "launch stitch_app..."

./stitch_app $CONFIG_FILE

echo "stitch_app exit"

echo "开始绘制 timing 图..."

python3 scripts/plot_timing.py