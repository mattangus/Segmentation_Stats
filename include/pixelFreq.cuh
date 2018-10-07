#pragma once

#include "helpers.cuh"
#include "stat.hpp"
#include "common.cuh"
#include "tensor.cuh"
#include "types.cuh"

#include <cuda.h>
#include <mutex>
#include <experimental/filesystem>
#include <sstream>

namespace fs = std::experimental::filesystem;

namespace cudaKernels
{
    __global__
    void accPixelFreq(unsigned char* gpu_im, long long* gpuFreqs , int h, int w, int d, int maxClass)
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

class pixelFreq : public stat
{
private:
    std::unordered_map<std::thread::id, std::shared_ptr<tensorInt64>> gpuFreqs;
    std::mutex writeLock;
    std::vector<long long> cpuRes;
public:
    int h,w,d;
    static const int maxClass = 20;
    std::shared_ptr<tensorInt64> gpuRes;
    pixelFreq()
    {
        this->name = "pixelFreq";
    }
    ~pixelFreq()
    {
        // tensor destructor will free memory
        // gpuErrchk( cudaFree(gpuRes->data) );
        // for(auto& f : gpuFreqs)
        //     gpuErrchk( cudaFree(f.second->data) );
        
    }
    void accumulate(cudnnHandle_t& cudnn, std::shared_ptr<tensorUint8>& gpuIm)
    {
        std::thread::id tId = std::this_thread::get_id();
        if(gpuIm->d != 1)
        {
            throw std::runtime_error("Cannot accumulate frequency with more than 1 depth dim. got: " + gpuIm->d);
        }
        if(gpuFreqs.count(tId) <= 0)
        {
            this->h = gpuIm->h; this->w = gpuIm->w; this->d = gpuIm->d;
            std::shared_ptr<tensorInt64> tempGpu(new tensorInt64(cudnn, h, w, d, maxClass, true));
            tempGpu->set(0);
            std::lock_guard<std::mutex> guard(writeLock);
            gpuFreqs[tId] = tempGpu;
        }
        if(this->h != gpuIm->h || this->w != gpuIm->w || this->d != gpuIm->d)
            throw std::runtime_error("Cannot handle different sized images");
        
        dim3 blockDim(16,16);
		dim3 blocks((w/blockDim.x)+1, (h/blockDim.y)+1); // blocks running on core
        cudaKernels::accPixelFreq<<<blocks, blockDim>>>(gpuIm->data, gpuFreqs[tId]->data, h, w, d, maxClass);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
    }
    void finalize(cudnnHandle_t& cudnn)
    {
        //nothing to finialize
    }
    void merge(cudnnHandle_t& cudnn)
    {
        gpuRes = std::shared_ptr<tensorInt64>(new tensorInt64(cudnn, h, w, d, maxClass, true));
        gpuRes->set(0);
        for (auto & it : gpuFreqs) {
            //add(gpuRes->data, it.second->data, gpuRes->data, h, w, maxClass);
            *gpuRes += *(it.second);
        }
        cpuRes = gpuRes->toCpu();
    }

    void save(std::string outputFolder)
    {
        fs::path base = outputFolder;
        base = base / name;
        fs::create_directories(base);
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
                cv::imwrite(ss.str(), classes[i]);
            }
        }
    }
};