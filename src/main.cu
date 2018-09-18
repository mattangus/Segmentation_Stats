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
#include <opencv2/opencv.hpp>

#include "helpers.cuh"
#include "frequency.cuh"
#include "statManager.hpp"

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

void processImage(std::string image_path, std::string output_path, std::string depth_path, float cutOff, cv::Mat kernel)
{
	// // output program input state
	// std::cout << "reading from path " << image_path << "..." << endl;
	// std::cout << "output path " << output_path << "..." << endl;
	// //std::cout << "depth path " << depth_path << "..." << std::endl;
	
	// // read image data in bgr then copy to standard array
	// cv::Mat im;
	// while(!im.data)
	// {
	// 	im = imread(image_path, CV_LOAD_IMAGE_COLOR);
	// 	if(!im.data) //only be able to parse if IEND chunk is found (i.e. transer complete)
	// 		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	// }

	// vector<float> depthData;
	// {//read depth
	// 	std::ifstream depthFile(depth_path.c_str(), std::ios::binary | std::ios::in);
		
	// 	if(depthFile.good())
	// 	{
	// 		while (!depthFile.eof())
	// 		{
	// 			float temp;
	// 			depthFile.read((char*)&temp, sizeof(temp));
	// 			depthData.push_back(temp);
	// 		}
	// 	}
	// 	else
	// 	{
	// 		for(int i = 0; i < im.rows*im.cols; i++)
	// 		{
	// 			depthData.push_back(10);
	// 		}
	// 		cutOff = 5;
	// 	}
	// }

	// /*
	// //view image read in for debugging
	// vector<uint8_t> depthImg;
	// for (int i = 0; i < depthData.size(); i++)
	// {
	// 	depthImg.push_back(255 - (depthData[i] * 255 / 300));
	// }

	// cv::Mat depthMat(600, 800, CV_8U, &depthImg[0]);
	// cv::imshow("test", depthMat);
	// cv::waitKey(0);
	// std::cout << "got " << depthData.size() << "depth value" << std::endl;
	// */

	// /**
	//  * CALL THE CUDA INTERFACE FUNC
	//  */
	// int start_s = clock();
	// cv::Mat output_mat = run_interpolation(im, depthData, cutOff, COLOURS_RAW);

	// std::vector<cv::Mat> bgr;
	// cv::split(output_mat, bgr);
	// auto veg = cv::Scalar(156,41,156);

	// cv::Mat mask = (bgr[0] == veg[0]) & (bgr[1] == veg[1]) & (bgr[2] == veg[2]);

	// cv::morphologyEx(mask, mask, MORPH_CLOSE, kernel);
	// output_mat.setTo(veg, mask);
	// int stop_s= clock();
	
	// /** ** **/
	// fs::create_directories(fs::path(output_path).parent_path());
	// std::vector<unsigned char> buf;
	// cv::imencode(fs::path(output_path).extension().string(), output_mat, buf);
	// //cv::imwrite(output_path + "t", output_mat);
	// {
	// 	std::ofstream ofs(output_path + "t", std::ofstream::binary);
	// 	ofs.write((const char*)&buf[0], buf.size());
	// }
	// fs::rename(output_path + "t", output_path);

	// // display execution time of the kernel function
	// cout << "done in " << (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000 << "msec!" << endl;
}

void processLoop(std::vector<std::string>* toProcess, std::string* base_path, int device, std::vector<std::shared_ptr<stat>>* stats)
{
	//each process loop has it's own thread!

	gpuErrchk( cudaSetDevice(device) );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	unsigned char* gpuIm;
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
			gpuErrchk( cudaMalloc((void**) &gpuIm, imSize) );
			inited = true;
		}

		gpuErrchk( cudaMemcpy(gpuIm, bgr[0].ptr(), imSize, cudaMemcpyHostToDevice) );

		for(auto s : *stats)
		{
			s->accumulate(gpuIm, h, w, 1);
		}

		// std::string outImgPath = std::regex_replace(curPath, std::regex(base_path), output_path);
		// processImage(curPath, outImgPath, "fake", 20, kernel);
	}

	for(auto s : *stats)
	{
		s->finalize();
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
	std::string base_path, depth_path;
	std::string ending = "png";
	int numThread = 8;
	std::vector<int> availDevice = {0}; //static max number because this uses a lot of GPU, so only one per GPU
	statManager manager;
	for (int i = 0; i < argc; i++)
	{
		if (strcmp(argv[i], "-i") == 0)
			base_path = argv[i+1];
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
	}

	std::cout << "base path: '" << base_path << "'" << std::endl;
	std::cout << "num thread: " << numThread << std::endl;
	std::cout << "using (";
	for(auto& v : availDevice)
		std::cout << v << ", ";
	std::cout << ") gpus" << std::endl;

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

	for(auto s : stats)
	{
		s->merge();
		s->viz();
	}

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
	float sec = ((float)duration.count()/1000.0f);
	std::cout << "finished in " << sec << " seconds (" << (toProcess.size()/sec) << " im/sec)" << std::endl;
}
