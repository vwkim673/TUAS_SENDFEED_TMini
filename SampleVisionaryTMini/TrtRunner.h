#pragma once

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // kINFO is very chatty; use kWARNING or lower for normal use
        if (severity <= Severity::kWARNING) {
            printf("[TRT] %s\n", msg);
        }
    }
};

struct TrtTensor {
    std::string name;
    nvinfer1::DataType dtype{};
    nvinfer1::Dims dims{};
    size_t bytes = 0;

    void* dptr = nullptr;              // device buffer
    std::vector<uint8_t> host;         // host buffer for outputs (or input if you want)
};

class TrtRunner {
public:
    TrtRunner();
    ~TrtRunner();

    // Load serialized engine (.engine)
    bool loadEngine(const std::string& enginePath);

    // For dynamic shapes: set input shape before allocating buffers / running
    // Example: setInputShape("images", Dims4{1,3,640,640})
    bool setInputShape(const std::string& inputName, const nvinfer1::Dims& dims);

    // Allocate GPU buffers based on current context shapes.
    // Call after setInputShape for dynamic engines.
    bool allocateIO();

    // Run inference.
    // Provide input as raw bytes (e.g., float32 tensor in NCHW).
    // inputBytes must match the input tensor size in bytes.
    bool infer(const std::string& inputName, const void* inputData, size_t inputBytes);

    // Access output tensors after infer()
    const std::vector<TrtTensor>& outputs() const { return m_outputs; }

    // Convenience getters
    std::vector<std::string> inputNames() const { return m_inputNames; }
    std::vector<std::string> outputNames() const { return m_outputNames; }

private:
    static std::vector<char> readFile(const std::string& path);
    static size_t elementSize(nvinfer1::DataType t);
    static size_t volume(const nvinfer1::Dims& d);
    static bool isDynamic(const nvinfer1::Dims& d);

    bool refreshTensorShapesAndSizes();

private:
    TrtLogger m_logger;

    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

    cudaStream_t m_stream = nullptr;

    std::vector<std::string> m_inputNames;
    std::vector<std::string> m_outputNames;

    // we keep one input tensor record + all output tensor records
    std::unordered_map<std::string, TrtTensor> m_allTensors;
    std::vector<TrtTensor> m_outputs; // snapshot view for easy iteration
};
#pragma once
