#pragma once

#include "helpers.cuh"
#include "stat.hpp"
#include "common.cuh"

#include <cuda.h>

namespace cudaKernels
{
    __global__
    void accFreq(unsigned char* gpu_im, long long* gpuFreqs , int h, int w, int d, int maxClass)
    {
        const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

        if (x >= w || y >= h) return;

        int ind = (y*w + x)*d;

        gpuFreqs[gpu_im[ind]]++;
    }
} //cudaKernels

class frequency : public stat
{
private:
    static const int maxClass = 256;
    std::unordered_map<std::thread::id, long long*> gpuFreqs;
    int h,w,d;
    long long* cpuRes;
public:
    frequency()
    {
        this->name = "frequency";
    }
    ~frequency(){ }
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
            ///int imSize = h*w*d*sizeof(long long)*maxClass;
            int imSize = sizeof(long long)*maxClass;
            long long* tempGpu;
            gpuErrchk( cudaMalloc((void**) &tempGpu, imSize) );
            gpuFreqs[tId] = tempGpu;
        }
        if(this->h != h || this->w != w || this->d != d)
            throw std::runtime_error("Cannot handle different sized images");
        
        dim3 threads(16,16);
		dim3 blocks((w/threads.x)+1, (h/threads.y)+1); // blocks running on core
        cudaKernels::accFreq<<<blocks, threads>>>(gpuIm, gpuFreqs[tId], h, w, d, maxClass);
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
        // int imSize = h*w*d*sizeof(long long)*maxClass;
        int imSize = sizeof(long long)*maxClass;
        gpuErrchk( cudaMalloc((void**) &tempGpu, imSize) );
        for (auto & it : gpuFreqs) {
            add(tempGpu, it.second, tempGpu, maxClass, 1, 1);
        }
        cpuRes = new long long[maxClass];
        gpuErrchk( cudaMemcpy(cpuRes, tempGpu, imSize, cudaMemcpyDeviceToHost) );
    }

    void viz()
    {
        for(int i = 0; i < maxClass; i++)
        {
            if(cpuRes[i] != 0)
                std::cout << "class " << i << ": " << cpuRes[i] << std::endl;
        }
        // std::vector<cv::Mat> classes(maxClass);
        // long double numPix = h*w;
        // long double* tempRes = new long double[maxClass];
        // for (int i = 0; i < maxClass; i++)
        // {
        //     tempRes[i] = cpuRes[i] / numPix;
        // }
        // for (int i=0; i < (int)classes.size(); i++) 
        // {
        //     classes[i] = cv::Mat(h, w, CV_64F, tempRes+i*h*w*sizeof(long double));
        //     double min, max;
        //     cv::minMaxLoc(classes[i], &min, &max);
        //     std::cout << min << "," << max << std::endl;
        //     cv::imshow(std::to_string(i), classes[i]);
        //     cv::waitKey();
        // }
    }
};