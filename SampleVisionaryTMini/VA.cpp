
#include <cuda_runtime.h>
#include "VA.h"

void writeVectorToFile(const std::vector<float>& data, const std::string& filename) {
	std::ofstream outFile(filename); // Open file for writing
	if (!outFile) {
		std::cerr << "Error opening file: " << filename << std::endl;
		return;
	}

	for (float value : data) {
		outFile << value << "\n"; // Write each float on a new line
	}

	outFile.close(); // Close the file
	std::cout << "Data successfully written to " << filename << std::endl;
}

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
	return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}
cv::Mat CreateLetterbox(cv::Mat img, cv::Size sz, cv::Scalar color,
	float& ratio, cv::Point& diff, cv::Point& diff2,
	bool autoSize = false, bool scaleFill = false, bool scaleup = true)
{
	cv::Mat newImage = cv::Mat();
	cv::cvtColor(img, newImage, cv::ColorConversionCodes::COLOR_BGR2RGB);

	ratio = std::min((float)sz.width / newImage.cols, (float)sz.height / newImage.rows);
	if (!scaleup)
	{
		ratio = std::min(ratio, 1.0f);
	}
	auto newUnpad = cv::Size((int)round(newImage.cols * ratio),
		(int)round(newImage.rows * ratio));

	//Width and height of needed border.
	auto dW = sz.width - newUnpad.width;
	auto dH = sz.height - newUnpad.height;

	auto temp_dW = dW / 2;
	auto temp_dH = dH / 2;

	auto tensor_ratio = sz.height / (float)sz.width;
	auto input_ratio = img.rows / (float)img.cols;
	if (autoSize && tensor_ratio != input_ratio)
	{
		dW %= 32;
		dH %= 32;
	}
	else if (scaleFill)
	{
		dW = 0;
		dH = 0;
		newUnpad = sz;
	}
	auto dW_h = (int)round((float)dW / 2);
	auto dH_h = (int)round((float)dH / 2);

	auto dw2 = 0;
	auto dh2 = 0;
	if (dW_h * 2 != dW)
	{
		dw2 = dW - dW_h * 2;
	}
	if (dH_h * 2 != dH)
	{
		dh2 = dH - dH_h * 2;
	}

	if (newImage.cols != newUnpad.width || newImage.rows != newUnpad.height)
	{
		cv::resize(newImage, newImage, newUnpad);
	}
	cv::copyMakeBorder(newImage, newImage, dH_h + dh2, dH_h,
		dW_h + dw2, dW_h, cv::BorderTypes::BORDER_CONSTANT, color);

	diff = cv::Point(temp_dW, temp_dH);
	diff2 = cv::Point(dw2, dh2);
	return newImage;
}

bool once = true;
auto VA::load_onnx_model(std::wstring onnx_path) -> bool
{
	const int64_t batchSize = 1;

#pragma region onnx session initialize
	onnx_env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "VA");

	onnx_so_ = Ort::SessionOptions();
	onnx_so_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
	onnx_so_.SetIntraOpNumThreads(1);
	onnx_so_.SetInterOpNumThreads(1);
	onnx_so_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	// Safe CUDA options
	OrtCUDAProviderOptions cuda_options;
	cuda_options.device_id = 0;
	cuda_options.arena_extend_strategy = 0;
	//cuda_options.gpu_mem_limit = size_t(512) * 1024 * 1024;
	cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
	cuda_options.do_copy_in_default_stream = 1;

	onnx_so_.AppendExecutionProvider_CUDA(cuda_options);

	//auto rtrnAEP = OrtSessionOptionsAppendExecutionProvider_CUDA(onnx_so_, 0);
	//if (rtrnAEP == NULL) printf("CUDA Launch Failed: fall back to CPU inference\n");
	onnx_session_ = Ort::Session(onnx_env_, onnx_path.c_str(), onnx_so_);
	printf("ONNX model loaded successfully from %ls\n", onnx_path.c_str());
#pragma endregion

	Ort::AllocatorWithDefaultOptions allocator;

	size_t numInputNodes = onnx_session_.GetInputCount();
	size_t numOutputNodes = onnx_session_.GetOutputCount();

	auto inputNameAllocated = onnx_session_.GetInputNameAllocated(0, allocator);
	std::string inputName = inputNameAllocated.get();

	Ort::TypeInfo inputTypeInfo = onnx_session_.GetInputTypeInfo(0);
	auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

	ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();

	inputDims_ = inputTensorInfo.GetShape();
	//printf("Input Shape : %d, %d, %d, %d\n", inputDims_.at(0), inputDims_.at(1), inputDims_.at(2), inputDims_.at(3));
	if (inputDims_.at(0) == -1)
	{
		//std::cout << "Got dynamic batch size. Setting input batch size to " << batchSize << "." << std::endl;
		inputDims_.at(0) = batchSize;
	}

	auto outputNameAllocated = onnx_session_.GetOutputNameAllocated(0, allocator);
	std::string outputName = outputNameAllocated.get();
	//printf("Output Name : %s\n", outputName.c_str());

	Ort::TypeInfo outputTypeInfo = onnx_session_.GetOutputTypeInfo(0);
	auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

	ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();

	outputDims_ = outputTensorInfo.GetShape();
	//printf("outputdims size: %d\n", outputDims_.size());
	//printf("Output Shape : %d, %d\n", outputDims_.at(0), outputDims_.at(1));// , outputDims_.at(2), outputDims_.at(3));
	if (outputDims_.at(0) == -1)
	{
		std::cout << "Got dynamic batch size. Setting output batch size to " << batchSize << "." << std::endl;
		outputDims_.at(0) = batchSize;
		outputDims_.at(1) = 70 * batchSize;
		//outputDims_.at(1) = 35 * batchSize; // at max 4. 
	}
	printf("Modified Output Shape : %d, %d\n", outputDims_.at(0), outputDims_.at(1));// , outputDims_.at(2), outputDims_.at(3));

	inputTensorSize_ = vectorProduct(inputDims_);
	outputTensorSize_ = vectorProduct(outputDims_);
	printf("Input Tensor Size : %d\n", inputTensorSize_);
	printf("Output Tensor Size : %d\n", outputTensorSize_);

	inputNames_ = std::vector<std::string>{ inputName };
	//printf("Input Name : %s\n", inputNames_.at(0).c_str());
	outputNames_ = std::vector<std::string>{ outputName };
	//printf("Output Name : %s\n", outputNames_.at(0).c_str());	

	memoryInfo_ = Ort::MemoryInfo::CreateCpu(
		OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
	
	return true;
}
auto VA::onnx_inference(cv::Mat image, std::vector<rectangle_info>& det_outs) -> bool
{
	//cv::Mat imgColor;
	//this gets 8UC1 to 8UC3 "BGR" image.
	//cv::cvtColor(image, imgColor, cv::COLOR_GRAY2BGR);

	auto imgSize = cv::Size(640, 640);
	auto padColor = cv::Scalar(144, 144, 144);

	auto ratio = 0.0f;
	auto diff1 = cv::Point();
	auto diff2 = cv::Point();

	bool isAuto = false;
	if (image.cols <= imgSize.width || image.rows <= imgSize.height) isAuto = false;

	auto letterimg = CreateLetterbox(image, imgSize, padColor, ratio, diff1, diff2);
	//std::cout << std::to_string(letterimg.cols) << " , " << std::to_string(letterimg.rows) << std::endl;
	//printf("letterimg data type: %d\n", letterimg.type());

	cv::Mat preprocessedImage = cv::Mat();
	letterimg.convertTo(preprocessedImage, CV_32F, (float)(1.0f / 255.0f));

	//std::cout << "Original (8-bit): " << letterimg(cv::Rect(0, 0, 3, 3)) << std::endl;
	//std::cout << "Converted (32-bit float): " << preprocessedImage(cv::Rect(0, 0, 3, 3)) << std::endl;

	//printf("preprocessedImage data size: %d,%d,%d \n", preprocessedImage.rows, preprocessedImage.cols, preprocessedImage.channels());
	//printf("preprocessedImage data type: %d\n", preprocessedImage.type());
	//printf("preprocessedImage data size: %d\n", preprocessedImage.total());
	
	//cv::imshow("test", preprocessedImage);
	//cv::waitKey(1);

	inputTensorValues_ = std::vector<float>(inputTensorSize_);
	
	std::copy(preprocessedImage.begin<float>(),
		preprocessedImage.end<float>(),
		inputTensorValues_.begin());

	/*
	std::copy(preprocessedImage.begin<float>(),
		preprocessedImage.end<float>(),
		inputTensorValues_.begin());

	std::copy(preprocessedImage.begin<float>(),
		preprocessedImage.end<float>(),
		inputTensorValues_.begin() + preprocessedImage.total());

	std::copy(preprocessedImage.begin<float>(),
		preprocessedImage.end<float>(),
		inputTensorValues_.begin() + preprocessedImage.total() * 2);
	*/
	outputTensorValues_ = std::vector<float>(outputTensorSize_);

	inputTensors_.clear();
	outputTensors_.clear();

	inputTensors_.push_back(Ort::Value::CreateTensor<float>(
		memoryInfo_, inputTensorValues_.data(), inputTensorSize_, inputDims_.data(),
		inputDims_.size()));

	outputTensors_.push_back(Ort::Value::CreateTensor<float>(
		memoryInfo_, outputTensorValues_.data(), outputTensorSize_,
		outputDims_.data(), outputDims_.size()));

	//Ort::Value outputTensor = Ort::Value::CreateTensor<float>(
	//	memoryInfo_,
	//	nullptr, 0,  // Let ORT allocate output buffer
	//	nullptr, 0); // No shape declared

	//outputTensors_.clear();
	//outputTensors_.push_back(std::move(outputTensor));

	const char* tempInputName = inputNames_.at(0).c_str();
	const char* tempoutputName = outputNames_.at(0).c_str();

	std::vector<const char*> inputNamess_{ tempInputName };
	std::vector<const char*> outputNamess_{ tempoutputName };

	printf("Input Name : %s\n", inputNamess_.at(0));

	std::cout << "inputTensorSize_: " << inputTensorSize_ << std::endl;
	std::cout << "inputTensorValues_.size(): " << inputTensorValues_.size() << std::endl;

	printf("Output Name : %s\n", outputNamess_.at(0));

	auto inference_time_start = std::chrono::high_resolution_clock::now();
	try
	{
		onnx_session_.Run(Ort::RunOptions{ nullptr }, inputNamess_.data(),
			inputTensors_.data(), 1, outputNamess_.data(),
			outputTensors_.data(), 1);
		//inference_exceptions_ = false;

		//cudaDeviceSynchronize();
	}
	catch (std::exception& ex)
	{
		printf(("Exception occurred during onnx inference: " + std::string(ex.what()) + "\n").c_str());
		//inference_exceptions_ = true;
	}
	catch (...)
	{
		printf("Exception occurred during onnx inference.");
		//inference_exceptions_ = true;
	}
	auto inference_time_end = std::chrono::high_resolution_clock::now();
	auto inference_dur = std::chrono::duration_cast<std::chrono::milliseconds>(inference_time_end - inference_time_start).count();
	//printf("onnx inference time : " + std::to_string(inference_dur) + std::string("ms");

	printf(("onnx inference time : " + std::to_string(inference_dur) + std::string("ms\n")).c_str());

	std::vector<rectangle_info> out_obj{};
	int bbx_count = 0;
	printf((std::string("size of output: ") + std::to_string(outputTensorValues_.size()) + std::string("\n")).c_str());
	for (int i = 0; i < outputTensorValues_.size(); i += 7)
	{
		//printf((std::string("det: ") + std::to_string(outputTensorValues_[i]) + std::string(" ") + std::to_string(outputTensorValues_[i+1]) + std::string(" ") + std::to_string(outputTensorValues_[i + 2]) + std::string(" ") + std::to_string(outputTensorValues_[i + 3]) + std::string(" ") + std::to_string(outputTensorValues_[i + 4]) + std::string(" ") + std::to_string(outputTensorValues_[i + 5]) + std::string(" ") + std::to_string(outputTensorValues_[i + 6]) + std::string(" ") + std::string("\n")).c_str());
		auto det_prob = int(outputTensorValues_[i + 6] * 100);
		//printf((std::string("det_prob: ") + std::to_string(det_prob) + std::string("\n")).c_str());
		if (det_prob > 0)
		{
			//printf("Anything here!?\n");
			auto r_x_min = (outputTensorValues_[i + 1] - diff1.x) / ratio;
			auto r_y_min = (outputTensorValues_[i + 2] - diff1.y) / ratio;
			auto r_x_max = (outputTensorValues_[i + 3] - diff1.x) / ratio;
			auto r_y_max = (outputTensorValues_[i + 4] - diff1.y) / ratio;

			auto outs = rectangle_info();

			outs.batchNum = outputTensorValues_[i];
			outs.ClassInfo = int(outputTensorValues_[i + 5]);

			outs.X = r_x_min;
			outs.Y = r_y_min;
			outs.W = r_x_max - r_x_min;
			outs.H = r_y_max - r_y_min;

			outs.Center_X = outs.X + int(outs.W / 2);
			outs.Center_Y = outs.Y + int(outs.H / 2);

			outs.Prob = int(outputTensorValues_[i + 6] * 100);

			det_outs.push_back(outs);
			//outs[bbx_count].Time = stopwatch.ElapsedMilliseconds;

			bbx_count++;

			//printf((std::string("ClassInfo: ") + std::to_string(outs.ClassInfo) + " Center X: " + std::to_string(outs.Center_X) + " Center Y: " + std::to_string(outs.Center_Y) + " Prob: " + std::to_string(outs.Prob) + "\n").c_str());
		}
	}


	return true;
}