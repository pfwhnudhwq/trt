#include "YoloTrt.h"

#define MAX_WORKSPACE (1 << 30)
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
    builder->setMaxBatchSize(1);
    builder->setMaxWorkspaceSize(MAX_WORKSPACE);

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

YoloTrt* GetYoloTrtObject(std::string path)
{
    return (new YoloTrt(path));
}
YoloTrt::~YoloTrt()
{
    engine->destroy();
    context->destroy();
}
YoloTrt::YoloTrt(std::string path)
{
    //load tensorrt engine
    gLogInfo<<"loading tensorrt engine from "<<path<<std::endl;
    // 从cached_model.bin文件中读取序列化的结果
    std::ifstream fin(path);
    IRuntime* runtime = createInferRuntime(gLogger.getTRTLogger());
    // 将文件中的内容读取至cached_engine字符串
    std::string cached_engine = "";
    while (fin.peek() != EOF){ // 使用fin.peek()防止文件读取时无限循环
        std::stringstream buffer;
        buffer << fin.rdbuf();
        cached_engine.append(buffer.str());
    }
    fin.close();
    engine =runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(),nullptr);
    if (!engine)
    {
	gLogInfo<<"engine error!."<<std::endl;
	return;
    }
    context=engine->createExecutionContext();
    gLogInfo<<"loading success!"<<std::endl;

    //malloc device memory
    int nbBindings=engine->getNbBindings();
    assert(nbBindings>=2);
    buffers.reserve(nbBindings);
    bindingIdxOutput.reserve(nbBindings-1);
    for(int i=0;i<nbBindings;i++)
    {
        Dims dims=engine->getBindingDimensions(i);
        DataType dtype=engine->getBindingDataType(i);
        if(engine->bindingIsInput(i))
	{
	    bindingIdxInput=i;
	    input_size=dims.d[0]*dims.d[1]*dims.d[2]*sizeof(dtype);
	    buffers[i]=safeCudaMalloc(input_size);
	}
        else
        {
            bindingIdxOutput.push_back(i);
	    int64_t size=dims.d[0]*dims.d[1]*dims.d[2];
	    eltCount+=size;
            buffers[i]=safeCudaMalloc(size*sizeof(dtype));
        }
    }
}
float* YoloTrt::execute(float* inputs)
{
    gLogInfo<<"copying data from host to device..."<<std::endl;
    //memcpy inputs from host to device	
    CHECK(cudaMemcpy(buffers[bindingIdxInput],inputs,input_size,cudaMemcpyHostToDevice));
    //inference
    context->execute(1,&buffers[0]);
    //malloc host output memory
    float* outputs=new float[eltCount];
    float* temp=outputs;
    for(int i=0;i<bindingIdxOutput.size();i++)
    {
        int index=bindingIdxOutput[i];
        Dims dims=engine->getBindingDimensions(index);
        DataType dtype=engine->getBindingDataType(index);
	int64_t count=dims.d[0]*dims.d[1]*dims.d[2];
	CHECK(cudaMemcpy(temp,buffers[index],count*sizeof(dtype),cudaMemcpyDeviceToHost));
	temp+=count;
    }
    return outputs;
}
void* YoloTrt::safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    //CHECK(cudaMalloc(&deviceMem, memSize));
    CHECK(cudaMallocManaged(&deviceMem,memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}
