#!/usr/bin/env sh

./build/tools/caffe test \
	--model=****/MobileNetV2_deploy.prototxt \
	--weights=****/mobilenet_v2_new.caffemodel \
	--gpu=0 --iterations=2000
