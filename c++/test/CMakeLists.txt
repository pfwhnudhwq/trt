cuda_add_executable(generateEngine generateEngine.cpp)
target_link_libraries(generateEngine yolo_tensorrt)
target_link_libraries(generateEngine ${ONNX_PARSER} ${UFF_PARSER} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY} ${OpenCV_LIBS})

