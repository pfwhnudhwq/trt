#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>
#include "NvInfer.h"
#include "NvUffParser.h"

#include "NvUtils.h"
#include "../common/argsParser.h"

using namespace nvuffparser;
using namespace nvinfer1;

#include "../common/logger.h"
#include "../common/common.h"
samplesCommon::Args gArgs;
#define MAX_WORKSPACE (1 << 30)
void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

ICudaEngine* createUffEngine(const char* uffFile, int maxBatchSize,
                                      IUffParser* parser)
{
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);
    INetworkDefinition* network = builder->createNetwork();

#if 1
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
    {
        gLogError << "Failure while parsing UFF file" << std::endl;
        return nullptr;
    }
#else
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kHALF))
    {
        gLogError << "Failure while parsing UFF file" << std::endl;
        return nullptr;
    }
    builder->setFp16Mode(true);
#endif

    /* we create the engine */
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(MAX_WORKSPACE);
    builder->setFp16Mode(gArgs.runInFp16);
    builder->setInt8Mode(gArgs.runInInt8);

    if (gArgs.runInInt8)
    {
        samplesCommon::setAllTensorScales(network, 127.0f, 127.0f);
    }

    if(gArgs.useDLACore)
    {
        for(int i=0;i<network->getNbLayers();i++)
        {
            auto layer=network->getLayer(i);
            std::cout<<layer->getName()<<":"<<builder->canRunOnDLA(layer)<<std::endl;
        }
    }

    samplesCommon::enableDLA(builder, gArgs.useDLACore);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine)
    {
        gLogError << "Unable to create engine" << std::endl;
        return nullptr;
    }

    /* we can clean the network and the parser */
    network->destroy();
    builder->destroy();

    return engine;
}
ICudaEngine* createOnnxEngine(const char* onnxFile, int maxBatchSize)
{
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);
    INetworkDefinition* network = builder->createNetwork();
    auto parser=nvonnxparser::createParser(*network,gLogger.getTRTLogger());
    if (!parser->parseFromFile(onnxFile,static_cast<int>(gLogger.getReportableSeverity())))
    {
        gLogError << "Failure while parsing ONNX file" << std::endl;
        return nullptr;
    }

    /* we create the engine */
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(MAX_WORKSPACE);
    builder->setFp16Mode(gArgs.runInFp16);
    builder->setInt8Mode(gArgs.runInInt8);

    if (gArgs.runInInt8)
    {
        samplesCommon::setAllTensorScales(network, 127.0f, 127.0f);
    }

    if(gArgs.useDLACore)
    {
        for(int i=0;i<network->getNbLayers();i++)
        {
            auto layer=network->getLayer(i);
            std::cout<<layer->getName()<<":"<<builder->canRunOnDLA(layer)<<std::endl;
        }
    }

    samplesCommon::enableDLA(builder, gArgs.useDLACore);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine)
    {
        gLogError << "Unable to create engine" << std::endl;
        return nullptr;
    }

    /* we can clean the network and the parser */
    network->destroy();
    builder->destroy();

    return engine;
}


int main(int argc, char** argv)
{
    bool argsOK = samplesCommon::parseArgs(gArgs, argc, argv);

    auto fileName=locateFile("yolo.onnx",std::vector<std::string>{"."});
    gLogInfo<<fileName<<std::endl;

    int maxBatchSize = 1;
    //auto parser = createUffParser();
    /* Register tensorflow input */
    //parser->registerInput("input", Dims3(3, 160, 384), UffInputOrder::kNCHW);
    //parser->registerOutput("Softmax");
    
    auto creator = getPluginRegistry()->getPluginCreator("ResizeNearest", "001");
    if(!creator){
    	gLogInfo<<"could not find plugin!."<<std::endl;
    }
    //ICudaEngine* engine = createUffEngine(fileName.c_str(), 1, parser);
    ICudaEngine* engine=createOnnxEngine(fileName.c_str(),1);
    if (!engine)
    {
        gLogError << "Model load failed" << std::endl;
        return 1;
    }
    // 得到序列化的模型结果
    IHostMemory *serializedModel = engine->serialize();
    engine->destroy();
    // 设置保存文件的名称为cached_model.bin
    std::string cache_path = "cached_model.bin";
    std::ofstream serialize_output_stream;

    // 将序列化的模型结果拷贝至serialize_str字符串
    std::string serialize_str;
    serialize_str.resize( serializedModel->size() );
    memcpy((void*)serialize_str.data(), serializedModel->data(), serializedModel->size());

    // 将serialize_str字符串的内容输出至cached_model.bin文件
    serialize_output_stream.open(cache_path);
    serialize_output_stream << serialize_str;
    serialize_output_stream.close();

    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine1 = runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size(), nullptr);
    serializedModel->destroy();
    return 1;
}

