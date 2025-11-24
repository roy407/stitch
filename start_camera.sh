#!/bin/bash

# 默认相机数量
NUM_CAM=5

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num)
            NUM_CAM="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: ./start_camera.sh -n <相机数量>"
            exit 1
            ;;
    esac
done

# 根据数量选择配置文件
case "$NUM_CAM" in
    5)
        CONFIG_FILE="resource/hk5"
        ;;
    8)
        CONFIG_FILE="resource/hk8"
        ;;
    *)
        echo "错误: 不支持的相机数量: $NUM_CAM"
        echo "仅支持 5, 8"
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

echo "绘图完成"