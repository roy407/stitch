.PHONY: all build deploy clean

BUILD_DIR := build
DEPLOY_PACKAGE := build/deploy.tar.gz

# 默认目标
all: build

# 编译目标
build:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make

# 打包目标
deploy: build
	rm -f $(DEPLOY_PACKAGE)
	tar -czvf $(DEPLOY_PACKAGE) -C $(BUILD_DIR) my_stitch_app resource mediamtx
	@echo "Packing deploy.tar.gz done !"
# 清理构建
clean:
	rm -rf $(BUILD_DIR) $(DEPLOY_PACKAGE)
