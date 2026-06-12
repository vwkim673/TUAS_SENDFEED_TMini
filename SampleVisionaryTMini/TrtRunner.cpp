#include "TrtRunner.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

static void checkCuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error (" << what << "): " << cudaGetErrorString(e) << "\n";
        throw std::runtime_error("CUDA failure");
    }
}

TrtRunner::TrtRunner() {}

TrtRunner::~TrtRunner() {
    // Free device buffers
    for (auto& kv : m_allTensors) {
        if (kv.second.dptr) {
            cudaFree(kv.second.dptr);
            kv.second.dptr = nullptr;
        }
    }
    if (m_stream) {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }
}

std::vector<char> TrtRunner::readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    f.read(data.data(), size);
    return data;
}

size_t TrtRunner::elementSize(nvinfer1::DataType t) {
    switch (t) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF:  return 2;
    case nvinfer1::DataType::kINT8:  return 1;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kBOOL:  return 1;
#if NV_TENSORRT_MAJOR >= 9
    case nvinfer1::DataType::kUINT8: return 1;
    case nvinfer1::DataType::kFP8:   return 1; // varies; treat as 1 byte for buffer sizing
#endif
    default: return 0;
    }
}

size_t TrtRunner::volume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; i++) {
        v *= static_cast<size_t>(d.d[i]);
    }
    return v;
}

bool TrtRunner::isDynamic(const nvinfer1::Dims& d) {
    for (int i = 0; i < d.nbDims; i++) {
        if (d.d[i] < 0) return true;
    }
    return false;
}

bool TrtRunner::loadEngine(const std::string& enginePath) {
    auto engineData = readFile(enginePath);
    if (engineData.empty()) {
        std::cerr << "Failed to read engine: " << enginePath << "\n";
        return false;
    }

    m_runtime.reset(nvinfer1::createInferRuntime(m_logger));
    if (!m_runtime) return false;

    // IMPORTANT
    initLibNvInferPlugins(&m_logger, "");

    m_engine.reset(m_runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!m_engine) {
        std::cerr << "deserializeCudaEngine failed\n";
        return false;
    }

    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        std::cerr << "createExecutionContext failed\n";
        return false;
    }



    checkCuda(cudaStreamCreate(&m_stream), "cudaStreamCreate");

    // Discover IO tensor names (name-based API)
    m_inputNames.clear();
    m_outputNames.clear();
    m_allTensors.clear();
    m_outputs.clear();

    const int nbIOTensors = m_engine->getNbIOTensors();
    for (int i = 0; i < nbIOTensors; i++) {
        const char* nm = m_engine->getIOTensorName(i);
        auto mode = m_engine->getTensorIOMode(nm);

        if (mode == nvinfer1::TensorIOMode::kINPUT)  m_inputNames.emplace_back(nm);
        if (mode == nvinfer1::TensorIOMode::kOUTPUT) m_outputNames.emplace_back(nm);

        TrtTensor t;
        t.name = nm;
        t.dtype = m_engine->getTensorDataType(nm);
        t.dims = m_engine->getTensorShape(nm); // may be dynamic (-1)
        m_allTensors[t.name] = std::move(t);
    }
   
    // For dynamic engines, user should call setInputShape() then allocateIO().
    // For static engines, we can allocate right away.
    bool anyDynamic = false;
    for (auto& in : m_inputNames) {
        if (isDynamic(m_engine->getTensorShape(in.c_str()))) {
            anyDynamic = true;
            break;
        }
    }
    if (!anyDynamic) {
        if (!allocateIO()) return false;
    }
    return true;
}

bool TrtRunner::setInputShape(const std::string& inputName, const nvinfer1::Dims& dims) {
    if (!m_context) return false;

    // Set the shape on the context for dynamic inputs
    if (!m_context->setInputShape(inputName.c_str(), dims)) {
        std::cerr << "setInputShape failed for " << inputName << "\n";
        return false;
    }
    return true;
}

bool TrtRunner::refreshTensorShapesAndSizes() {
    // After setInputShape, context can resolve output shapes.
    // Use context->getTensorShape() for actual runtime sizes.
    for (auto& kv : m_allTensors) {
        const std::string& name = kv.first;
        auto& t = kv.second;
        t.dims = m_context->getTensorShape(name.c_str());
        if (isDynamic(t.dims)) {
            std::cerr << "Tensor still has dynamic dims unresolved: " << name << "\n";
            return false;
        }
        size_t elems = volume(t.dims);
        size_t bytes = elems * elementSize(t.dtype);
        if (bytes == 0) {
            std::cerr << "Unknown datatype sizing for tensor: " << name << "\n";
            return false;
        }
        t.bytes = bytes;
    }
    return true;
}

bool TrtRunner::allocateIO() {
    if (!m_engine || !m_context) return false;

    // Resolve shapes based on current context settings
    if (!refreshTensorShapesAndSizes()) {
        std::cerr << "refreshTensorShapesAndSizes failed. "
            "Did you forget setInputShape() for dynamic engines?\n";
        return false;
    }

    // Free any previous allocations
    for (auto& kv : m_allTensors) {
        if (kv.second.dptr) {
            cudaFree(kv.second.dptr);
            kv.second.dptr = nullptr;
        }
        kv.second.host.clear();
    }

    // Allocate GPU buffers and set tensor addresses
    for (auto& kv : m_allTensors) {
        auto& t = kv.second;
        checkCuda(cudaMalloc(&t.dptr, t.bytes), ("cudaMalloc " + t.name).c_str());

        // For outputs, we keep a host buffer too
        auto mode = m_engine->getTensorIOMode(t.name.c_str());
        if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            t.host.resize(t.bytes);
        }

        // Bind this tensor buffer to the context
        if (!m_context->setTensorAddress(t.name.c_str(), t.dptr)) {
            std::cerr << "setTensorAddress failed for tensor: " << t.name << "\n";
            return false;
        }
    }

    // Build outputs snapshot vector
    m_outputs.clear();
    for (const auto& outName : m_outputNames) {
        m_outputs.push_back(m_allTensors[outName]);
    }

    return true;
}

bool TrtRunner::infer(const std::string& inputName, const void* inputData, size_t inputBytes) {
    if (!m_context) return false;

    auto it = m_allTensors.find(inputName);
    if (it == m_allTensors.end()) {
        std::cerr << "Unknown input tensor name: " << inputName << "\n";
        return false;
    }

    TrtTensor& inT = it->second;
    if (inT.bytes != inputBytes) {
        std::cerr << "Input byte size mismatch for " << inputName
            << " expected=" << inT.bytes << " got=" << inputBytes << "\n";
        return false;
    }

    // H2D
    checkCuda(cudaMemcpyAsync(inT.dptr, inputData, inputBytes, cudaMemcpyHostToDevice, m_stream),
        "cudaMemcpyAsync H2D input");

    // Execute
    if (!m_context->enqueueV3(m_stream)) {
        std::cerr << "enqueueV3 failed\n";
        return false;
    }

    // D2H outputs
    for (const auto& outName : m_outputNames) {
        auto& outT = m_allTensors[outName];
        checkCuda(cudaMemcpyAsync(outT.host.data(), outT.dptr, outT.bytes,
            cudaMemcpyDeviceToHost, m_stream),
            "cudaMemcpyAsync D2H output");
    }

    checkCuda(cudaStreamSynchronize(m_stream), "cudaStreamSynchronize");

    // Refresh outputs snapshot (host buffers updated)
    m_outputs.clear();
    for (const auto& outName : m_outputNames) {
        m_outputs.push_back(m_allTensors[outName]);
    }

    return true;
}
