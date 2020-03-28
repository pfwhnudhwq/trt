#ifndef YOLOTRT_H
#define YOLOTRT_H

#include "NvOnnxParser.h"
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "NvInfer.h"

#include "NvUtils.h"

using namespace nvinfer1;

#include "logger.h"
#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cout << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

class YoloTrt{
public:
	//load tensorrt engine
	YoloTrt(std::string trtPath);
	~YoloTrt();
	//do inference
	float* execute(float* inputs);
private:
	void* safeCudaMalloc(size_t memSize);
	//inference context;
	ICudaEngine* engine;	
	IExecutionContext* context;
	//input index
	int bindingIdxInput;
	size_t input_size;
	//output indexs
	std::vector<int> bindingIdxOutput;
  int64_t eltCount;
	//output memory
	std::vector<void*> buffers;
};

extern YoloTrt* GetYoloTrtObject(std::string path);
#endif
