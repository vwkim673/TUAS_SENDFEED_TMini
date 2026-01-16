#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <sstream>
#include <fstream>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "GlobalVar.h"

#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>

class VA
{
public:
	VA() = default;

	auto load_onnx_model_new(std::wstring onnx_path) -> bool;
	auto load_onnx_model(std::wstring onnx_path) -> bool;
	auto onnx_inference(cv::Mat image, std::vector<rectangle_info>& det_outs) -> bool;

	private:
		std::string model_name_;

		Ort::Env onnx_env_;
		Ort::SessionOptions onnx_so_;
		Ort::RunOptions onnx_ro_;

		Ort::Session onnx_session_{ nullptr };

		std::vector<int64_t> inputDims_;
		size_t inputTensorSize_;
		std::vector<float> inputTensorValues_;

		std::vector<int64_t> outputDims_;
		size_t outputTensorSize_;
		std::vector<float> outputTensorValues_;

		Ort::MemoryInfo memoryInfo_{ nullptr };

		std::vector<std::string> inputNames_;
		std::vector<std::string> outputNames_;
		std::vector<Ort::Value> inputTensors_;
		std::vector<Ort::Value> outputTensors_;

};
