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
    frequency(std::shared_ptr<pixelFreq>& pixelFreqs, bool usePixelFreq) : pixelFreqs(pixelFreqs), usePixelFreq(usePixelFreq)
    {
        this->name = "frequency";
    }
    ~frequency() { }
    void accumulate(cudnnHandle_t& cudnn, std::shared_ptr<tensorUint8>& gpuIm)
    {
        pixelFreqs->accumulate(cudnn, gpuIm);
    }
    void finalize(cudnnHandle_t& cudnn)
    {
        //nothing to finialize
    }
    void merge(cudnnHandle_t& cudnn)
    {
        pixelFreqs->merge(cudnn);

        std::shared_ptr<tensorFloat64> gpuTemp(new tensorFloat64(pixelFreqs->gpuRes->cast<double>()));

        auto cpuTemp = gpuTemp->toCpu();
        // for(int i = 0; i < cpuTemp.size(); i++)
        // {
        //     if(cpuTemp[i] != 0)
        //     {
        //         int a = 0;
        //         int b = a;
        //     }
        // }

        // double* gpuTemp;
        // int h = pixelFreqs->h;
        // int w = pixelFreqs->w;
        // int d = pixelFreqs->d;
        // int maxClass = pixelFreqs->maxClass;
        // size_t size = h * w * d * maxClass;
        // gpuErrchk( cudaMalloc(&gpuTemp, size * sizeof(double)) );

        // cast(pixelFreqs->gpuRes, gpuTemp, size);
        std::shared_ptr<tensorFloat64> gpuMin(new tensorFloat64(gpuTemp->reduceMinAll()));
        std::shared_ptr<tensorFloat64> gpuMax(new tensorFloat64(gpuTemp->reduceMaxAll()));
        // double* gpuMin = reduceMinAll(cudnn, gpuTemp, h, w, maxClass);
        // double* gpuMax = reduceMaxAll(cudnn, gpuTemp, h, w, maxClass);

        std::shared_ptr<tensorFloat64> gpuRes(new tensorFloat64(gpuTemp->reduceSum({0, 1, 2})));

        // double* gpuRes = reduceSum(cudnn, gpuTemp, {0,1}, h, w, maxClass);

        cpuRes = gpuRes->toCpu();
        // gpuErrchk( cudaMemcpy(&cpuRes[0], gpuRes, maxClass*sizeof(double), cudaMemcpyDeviceToHost) );

        // std::vector<double> cpuTemp(h * w * d * maxClass);
        // gpuErrchk( cudaMemcpy(&cpuTemp[0], gpuTemp, h*w*d*maxClass*sizeof(double), cudaMemcpyDeviceToHost) );

        // gpuErrchk( cudaFree(gpuRes) );
        // gpuErrchk( cudaFree(gpuTemp) );
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
            std::cout  << i << ", [";
            for(size_t j = 0; j < 4; j++)
                std::cout << std::hex << (int)(((unsigned char*)&cpuRes[i])[j]) << ", ";
            std::cout << "]" << std::endl;
        }
    }
};