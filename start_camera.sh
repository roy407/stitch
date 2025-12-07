#!/bin/bash

# 默认相机数量
NUM_CAM=5

CAMERAS_DEBUG=OFF

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num)
            NUM_CAM="$2"
            shift 2
            ;;
        --debug)
            CAMERAS_DEBUG=ON
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: ./start_camera.sh [-n 数量] [--debug]"
            exit 1
            ;;
    esac
done

# 根据数量选择配置文件
case "$NUM_CAM" in
    5)
        CONFIG_FILE="resource/hk5"
        ;;
    10)
        CONFIG_FILE="resource/cam10"
        ;;
    *)
        echo "错误: 不支持的相机数量: $NUM_CAM"
        echo "仅支持 5, 10"
        exit 1
        ;;
esac

echo "调试宏 cameras_debug: $CAMERAS_DEBUG"
make build CAMERAS_DEBUG=$CAMERAS_DEBUG

echo "使用配置文件: $CONFIG_FILE"
echo "launch stitch_app..."

./build/stitch_app $CONFIG_FILE

echo "stitch_app exit"

echo "开始绘制 timing 图..."

python3 scripts/plot_timing.py