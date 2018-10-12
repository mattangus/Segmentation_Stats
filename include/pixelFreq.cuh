#pragma once

#include "helpers.cuh"
#include "stat.hpp"
#include "common.cuh"
#include "tensor.cuh"
#include "types.cuh"
#include "thread_map.hpp"

#include <cuda.h>
#include <experimental/filesystem>
#include <sstream>

namespace fs = std::experimental::filesystem;

namespace cudaKernels
{
    __global__
    void accPixelFreq(unsigned char* gpu_im, long long* gpuFreqs, int n, int h, int w, int maxClass)
    {
        const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
        // only support d = 1

        if (x >= w || y >= h) return;

        int ind = (y*w + x);
        unsigned char c = gpu_im[ind];
        if(c >= maxClass)
            c = maxClass - 1;
        ind = (c*h + y)*w + x;
        gpuFreqs[ind]++;
        
    }
} //cudaKernels

class pixelFreq : public stat
{
private:
    thread_map<std::shared_ptr<tensorInt64>> gpuFreqs;
    std::vector<long long> cpuRes;
public:
    int h,w,d;
    const int maxClass;
    std::shared_ptr<tensorInt64> gpuRes;
    pixelFreq(int maxClass) : maxClass(maxClass)
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
    void accumulate(cudaThreadCtx* ctx, tensorUint8& gpuIm, std::string& path)
    {
        if(gpuIm.d != 1)
        {
            throw std::runtime_error("Cannot accumulate frequency with more than 1 depth dim. got: " + gpuIm.d);
        }
        if(!gpuFreqs.hasData())
        {
            this->h = gpuIm.h; this->w = gpuIm.w; this->d = gpuIm.d;
            std::shared_ptr<tensorInt64> tempGpu(new tensorInt64(ctx, 1, h, w, maxClass, true));
            tempGpu->set(0);
            gpuFreqs.set(tempGpu);
        }
        if(!gpuIm.isCompatable(1,h,w,d))
            throw std::runtime_error("Cannot handle different sized images");
        
        dim3 blockDim(16,16);
		dim3 blocks((w/blockDim.x)+1, (h/blockDim.y)+1); // blocks running on core
        cudaKernels::accPixelFreq<<<blocks, blockDim, 0, ctx->stream>>>(gpuIm.getData(), gpuFreqs.get()->getData(), 1, h, w, maxClass);
        #ifdef SYNC_STREAM
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaStreamSynchronize(ctx->stream) );
        #endif
    }
    void finalize(cudaThreadCtx* ctx)
    {
        //nothing to finialize
    }
    void merge(cudaThreadCtx* ctx)
    {
        gpuRes = std::shared_ptr<tensorInt64>(new tensorInt64(ctx, 1, h, w, maxClass, true));
        gpuRes->set(0);
        for (auto & it : gpuFreqs.toList()) {
            //add(gpuRes->data, it.second->data, gpuRes->data, h, w, maxClass);
            *gpuRes += *(it);
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