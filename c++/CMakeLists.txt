cmake_minimum_required(VERSION 2.8.3)
project(tensorrt)

#set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")	# -std=gnu++11
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

#cuda
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

#tensorrt
find_path(TENSORRT_INCLUDE_DIR NvInfer.h)
find_library(TENSORRT_LIBRARY_INFER nvinfer)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
#include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/commom)

#uff
find_library(UFF_PARSER nvparsers)
message(STATUS "FOUND ${UFF_PARSER}")
#onnx
find_library(ONNX_PARSER nvonnxparser)
#opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_LIBS})

add_subdirectory(common)
add_subdirectory(test)
