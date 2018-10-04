//============================================================================
// Name        : main.cu
// Author      : Matt Angus
// Version     : 1.0.0
// Description : Entry point
//============================================================================

#include <exception>
#include <ctime>
#include <stdlib.h>
#include <experimental/filesystem>
#include <regex>
#include <thread>
#include <chrono>

#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <opencv2/opencv.hpp>

#include "helpers.cuh"
#include "pixelFreq.cuh"
#include "statManager.hpp"
#include "tensor.h"

namespace fs = std::experimental::filesystem;

bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

std::vector<std::string> GetImagesToProcess(std::string& inputPath, std::string& ending)
{
	std::vector<std::string> ret;
	for(auto& p: fs::recursive_directory_iterator(inputPath))
	{
		std::string curPath = p.path().string();
		bool regFile = fs::is_regular_file(p);
		if(regFile && hasEnding(curPath, ending))
		{
			ret.push_back(curPath);
		}
	}
	return ret;
}

void processLoop(std::vector<std::string>* toProcess, std::string* base_path, int device, std::vector<std::shared_ptr<stat>>* stats)
{
	//each process loop has it's own thread!
	cudnnHandle_t cudnn;
	gpuErrchk( cudnnCreate(&cudnn) );

	gpuErrchk( cudaSetDevice(device) );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	std::shared_ptr<tensor<unsigned char>> gpuIm;
	bool inited = false;

	int h, w;

	for(int i = 0; i < (int)toProcess->size(); i++)
	{
		std::string curPath = (*toProcess)[i];

		cv::Mat im;
		while(!im.data)
		{
			im = cv::imread(curPath, CV_LOAD_IMAGE_COLOR);
			if(!im.data) //only be able to parse if IEND chunk is found
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
		cv::Mat bgr[3]; //destination array
		cv::split(im,bgr); //split source
		h = im.rows;
		w = im.cols;
		int imSize = h*w*sizeof(unsigned char);
		if(!inited)
		{
			//gpuIm = std::shared_ptr<tensor<unsigned char>>(new tensor<unsigned char>(cudnn, 1, h, w, 1, true));
			gpuErrchk( cudaMalloc((void**) &gpuIm, imSize) );
			inited = true;
		}

		gpuErrchk( cudaMemcpy(gpuIm, bgr[0].ptr(), imSize, cudaMemcpyHostToDevice) );

		for(auto s : *stats)
		{
			s->accumulate(cudnn, gpuIm, h, w, 1);
		}

		// std::string outImgPath = std::regex_replace(curPath, std::regex(base_path), output_path);
		// processImage(curPath, outImgPath, "fake", 20, kernel);
	}
	gpuErrchk( cudaFree(gpuIm) );
	for(auto s : *stats)
	{
		s->finalize(cudnn);
	}
	// std::cout << "done" << std::endl;
}

template<typename T>
std::vector<std::vector<T>> SplitVector(const std::vector<T>& vec, size_t n)
{
    std::vector<std::vector<T>> outVec;

    size_t length = vec.size() / n;
    size_t remain = vec.size() % n;

    size_t begin = 0;
    size_t end = 0;

    for (size_t i = 0; i < std::min(n, vec.size()); ++i)
    {
        end += (remain > 0) ? (length + !!(remain--)) : length;

        outVec.push_back(std::vector<T>(vec.begin() + begin, vec.begin() + end));

        begin = end;
    }

    return outVec;
}

std::vector<int> parseDeviceList(std::string devList)
{
	std::vector<int> ret;
	std::vector<std::string> splitStr = split(devList, ',');
	for(std::string& s : splitStr)
	{
		ret.push_back(atoi(s.c_str()));
	}
	return ret;
}

/**
 * contains cuda specific initializations
 */
int main( int argc, char** argv )
{
	// grab the arguments
	std::string base_path, depth_path, output_path;
	std::string ending = "png";
	int numThread = 8;
	std::vector<int> availDevice = {0}; //static max number because this uses a lot of GPU, so only one per GPU
	statManager manager;
	for (int i = 0; i < argc; i++)
	{
		if (strcmp(argv[i], "-i") == 0)
			base_path = argv[i+1];
		if (strcmp(argv[i], "-o") == 0)
			output_path = argv[i+1];
		if (strcmp(argv[i], "-d") == 0)
			depth_path = argv[i + 1];
		if (strcmp(argv[i], "-e") == 0)
			ending = argv[i + 1];
		if (strcmp(argv[i], "-n") == 0)
			numThread = atoi(argv[i+1]);
		if (strcmp(argv[i], "-g") == 0)
			availDevice =  parseDeviceList(argv[i+1]);
		if (strcmp(argv[i], "-f") == 0)
			manager.addFrequency();
		if (strcmp(argv[i], "-p") == 0)
			manager.addPixelFrequency();
	}

	std::cout << "base path: '" << base_path << "'" << std::endl;
	std::cout << "output path: '" << output_path << "'" << std::endl;
	std::cout << "num thread: " << numThread << std::endl;
	std::cout << "using gpus (";
	for(auto& v : availDevice)
		std::cout << v << ", ";
	std::cout << ")" << std::endl;

	std::vector<std::shared_ptr<stat>> stats = manager.getStatList();

	std::vector<std::string> toProcess = GetImagesToProcess(base_path, ending);
	auto start = std::chrono::steady_clock::now();

	if(numThread > 1)
	{
		std::vector<std::thread> threads;
		std::vector<std::vector<std::string>> splitVals = SplitVector(toProcess, numThread);
		for(int i = 0; i < numThread && i < (int)splitVals.size(); i++)
		{
			int d = availDevice[i % availDevice.size()];
			threads.push_back(std::thread(processLoop, &splitVals[i], &base_path, d, &stats));
		}

		for(std::thread& t : threads)
		{
			t.join();
		}
	}
	else if(toProcess.size() > 0)
	{
		processLoop(&toProcess, &base_path, availDevice[0], &stats);
	}
	
	//main thread doesnt have a cudnn handle
	cudnnHandle_t cudnn;
	gpuErrchk( cudnnCreate(&cudnn) );

	for(auto s : stats)
	{
		s->merge(cudnn);
	}

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
	float sec = ((float)duration.count()/1000.0f);
	std::cout << "finished in " << sec << " seconds (" << (toProcess.size()/sec) << " im/sec)" << std::endl;

	start = std::chrono::steady_clock::now();
	for(auto s : stats)
	{
		std::cout << "===============STAT===============" << std::endl;
		s->save(output_path);
		std::cout << "==================================" << std::endl;
	}
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
	std::cout << "finished in " << sec << " seconds" << std::endl;
	std::cout << std::endl;
}
