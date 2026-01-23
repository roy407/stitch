#!/bin/bash

# 默认相机数量
CAM_JSON=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            CAM_JSON="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: ./start_camera.sh [-n 数量]"
            exit 1
            ;;
    esac
done

# 根据数量选择配置文件
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
    "cam_with_nowindow")
        CONFIG_FILE="resource/cam_with_nowindow"
        ;;
    *)
        echo "错误: 不支持的相机格式: $CAM_JSON"
        exit 1
        ;;
esac

make build

echo "使用配置文件: $CONFIG_FILE"
echo "launch stitch_app..."

./build/stitch_app $CONFIG_FILE

echo "stitch_app exit"

echo "开始绘制 timing 图..."

python3 scripts/plot_timing.py