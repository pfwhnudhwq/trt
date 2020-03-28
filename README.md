# trt
tensorrt accelerate yolo

cfg to onnx:
cd python/yolov3_onnx
python yolov3_to_onnx.py

onnx to tensorrt engine:
cd c++/build
cmake ..
make
cd ../bin
./generateEngine --fp16

