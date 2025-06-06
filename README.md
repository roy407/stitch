## 基于GPU的视频拼接项目

## 前置条件

1. nvidia driver >= 535

## 运行步骤如下所示：

1. make stitch_env  #下载所需镜像

2. make build       #程序编译

3. make deploy      #程序打包

## 注：如果只是需要跑代码，可以不进行make deploy这一步，可以make build之后直接cd 进入build目录，运行./stitch_app