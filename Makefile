.PHONY: all build deploy clean

BUILD_DIR := build
DEPLOY_PACKAGE := build/deploy.tar.gz

# 默认目标
all: build

stitch_env:
	docker run -it -v $(PWD):/opt/ffmpeg \
	-v /home/eric/mp4:/home/eric/mp4 \
	--network host --rm \
	--gpus all --runtime=nvidia \
	-e NVIDIA_VISIBLE_DEVICES=all \
	-e NVIDIA_DRIVER_CAPABILITIES=compute,video,utility,graphics \
	crpi-3l2tp5weim3sy2nu.cn-hangzhou.personal.cr.aliyuncs.com/nknk/stitch_env:v1.4 bash

# 编译目标
build:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make

# 打包目标
deploy: build
	rm -f $(DEPLOY_PACKAGE)
	tar -czvf $(DEPLOY_PACKAGE) -C $(BUILD_DIR) stitch_app resource mediamtx
	@echo "Packing deploy.tar.gz done !"
# 清理构建
clean:
	rm -rf $(BUILD_DIR) $(DEPLOY_PACKAGE)
