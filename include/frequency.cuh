#pragma once

#include "helpers.cuh"
#include "stat.hpp"
#include "common.cuh"

#include <cuda.h>
#include <mutex>
#include <experimental/filesystem>
#include <sstream>

namespace fs = std::experimental::filesystem;

namespace cudaKernels
{
    __global__
    void accFreq(unsigned char* gpu_im, long long* gpuFreqs , int h, int w, int d, int maxClass)
    {
        const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
        // only support d = 1

        if (x >= w || y >= h) return;

        int ind = (y*w + x);
        unsigned char cls = gpu_im[ind];
        ind = (y*w + x)*maxClass;
        if(cls < maxClass)
            gpuFreqs[ind + cls]++;
    }
} //cudaKernels

class frequency : public stat
{
private:
    static const int maxClass = 20;
    std::unordered_map<std::thread::id, long long*> gpuFreqs;
    std::mutex writeLock;
    int h,w,d;
    long long* cpuRes;
public:
    frequency()
    {
        this->name = "frequency";
    }
    ~frequency()
    {
        if(cpuRes)
            delete cpuRes;
        // for(long long* f : gpuFreqs)
        //     gpuErrchk( cudaFree(f) );
        
    }
    void accumulate(unsigned char* gpuIm, int h, int w, int d)
    {
        std::thread::id tId = std::this_thread::get_id();
        if(d != 1)
        {
            throw std::runtime_error("Cannot accumulate frequency with more than 1 depth dim");
        }
        if(gpuFreqs.count(tId) <= 0)
        {
            this->h = h; this->w = w; this->d = d;
            long imSize = h*w*d*sizeof(long long)*maxClass;
            //int imSize = sizeof(long long)*maxClass;
            long long* tempGpu;
            gpuErrchk( cudaMalloc((void**) &tempGpu, imSize) );
            gpuErrchk( cudaMemset(tempGpu, 0, imSize) );
            std::lock_guard<std::mutex> guard(writeLock);
            gpuFreqs[tId] = tempGpu;
        }
        if(this->h != h || this->w != w || this->d != d)
            throw std::runtime_error("Cannot handle different sized images");
        
        dim3 blockDim(16,16);
		dim3 blocks((w/blockDim.x)+1, (h/blockDim.y)+1); // blocks running on core
        cudaKernels::accFreq<<<blocks, blockDim>>>(gpuIm, gpuFreqs[tId], h, w, d, maxClass);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
    }
    void finalize()
    {
        //nothing to finialize
    }
    void merge()
    {
        long long* tempGpu;
        long imSize = h*w*d*sizeof(long long)*maxClass;
        //int imSize = sizeof(long long)*maxClass;
        gpuErrchk( cudaMalloc((void**) &tempGpu, imSize) );
        for (auto & it : gpuFreqs) {
            add(tempGpu, it.second, tempGpu, h, w, maxClass);
        }
        cpuRes = new long long[h*w*d*maxClass];
        gpuErrchk( cudaMemcpy(cpuRes, tempGpu, imSize, cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaFree(tempGpu) );
    }

    void save(std::string outputFolder)
    {
        fs::path base = outputFolder;
        base = base / name;
        fs::create_directories(base);
        long double numPix = h*w*d;
        std::vector<std::vector<unsigned char>> frames;
        std::vector<bool> hasValues;
        for (int c = 0; c < maxClass; c++)
        {
            std::vector<unsigned char> cur(h*w*d);
            long double maxVal = -1;
            for (int i = 0; i < h*w*d; i++)
            {
                long double temp = (long double)cpuRes[i*maxClass + c];
                if(temp > maxVal)
                    maxVal = temp;
            }
            for (int i = 0; i < h*w*d; i++)
            {
                long double temp = (long double)cpuRes[i*maxClass + c] / maxVal;
                cur[i] = (unsigned char)(temp*255.0);
            }
            hasValues.push_back(maxVal != 0);
            frames.push_back(cur);
        }
        std::vector<cv::Mat> classes(maxClass);
        for (int i=0; i < (int)classes.size(); i++) 
        {
            classes[i] = cv::Mat(h, w, CV_8UC1);
            memcpy(classes[i].data, frames[i].data(), frames[i].size()*sizeof(char));
            if(hasValues[i])
            {
                std::stringstream ss;
                ss << base.string() << "/" << i << ".png";
                cv::imshow(std::to_string(i), classes[i]);
                cv::imwrite(ss.str(), classes[i]);
            }
        }
        cv::waitKey();
    }
};