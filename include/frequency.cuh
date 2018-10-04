#pragma once

#include "helpers.cuh"
#include "stat.hpp"
#include "common.cuh"
#include "pixelFreq.cuh"

#include <cuda.h>
#include <mutex>
#include <experimental/filesystem>
#include <sstream>
#include <iostream>
#include <fstream>

namespace fs = std::experimental::filesystem;

class frequency : public stat
{
private:
    std::vector<double> cpuRes;
    std::shared_ptr<pixelFreq> pixelFreqs;
    bool usePixelFreq; // if true then pixel freqs should be saved as well
public:
    frequency(std::shared_ptr<pixelFreq> pixelFreqs, bool usePixelFreq) : pixelFreqs(pixelFreqs), usePixelFreq(usePixelFreq)
    {
        this->name = "frequency";
    }
    ~frequency() { }
    void accumulate(cudnnHandle_t& cudnn, unsigned char* gpuIm, int h, int w, int d)
    {
        pixelFreqs->accumulate(cudnn, gpuIm, h, w, d);
    }
    void finalize(cudnnHandle_t& cudnn)
    {
        //nothing to finialize
    }
    void merge(cudnnHandle_t& cudnn)
    {
        pixelFreqs->merge(cudnn);

        double* gpuTemp;
        int h = pixelFreqs->h;
        int w = pixelFreqs->w;
        int d = pixelFreqs->d;
        int maxClass = pixelFreqs->maxClass;
        size_t size = h * w * d * maxClass;
        gpuErrchk( cudaMalloc(&gpuTemp, size * sizeof(double)) );

        cast(pixelFreqs->gpuRes, gpuTemp, size);

        double* gpuMin = reduceMinAll(cudnn, gpuTemp, h, w, maxClass);
        double* gpuMax = reduceMaxAll(cudnn, gpuTemp, h, w, maxClass);

        std::vector<int> axes = {0,1};
        double* gpuRes = reduceSum(cudnn, gpuTemp, axes, h, w, maxClass);

        cpuRes = std::vector<double>(maxClass);
        gpuErrchk( cudaMemcpy(&cpuRes[0], gpuRes, maxClass*sizeof(double), cudaMemcpyDeviceToHost) );

        // std::vector<double> cpuTemp(h * w * d * maxClass);
        // gpuErrchk( cudaMemcpy(&cpuTemp[0], gpuTemp, h*w*d*maxClass*sizeof(double), cudaMemcpyDeviceToHost) );

        gpuErrchk( cudaFree(gpuRes) );
        gpuErrchk( cudaFree(gpuTemp) );
    }

    void save(std::string outputFolder)
    {
        if(usePixelFreq)
            pixelFreqs->save(outputFolder);
        
        fs::path base = outputFolder;
        fs::path outFile = base / (name + ".csv");

        std::ofstream csv(outFile);

        csv << "class,count" << std::endl;
        for(size_t i = 0; i < cpuRes.size(); i++)
        {
            csv << i << "," << cpuRes[i] << std::endl;
        }
    }
};