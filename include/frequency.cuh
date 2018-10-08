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

        tensorFloat64 gpuTemp = pixelFreqs->gpuRes->cast<double>();

        std::vector<double> cpuTemp = gpuTemp.toCpu();
        double maxV = -1000000;
        for(int i = 0; i < (int)cpuTemp.size(); i++)
        {
            if(cpuTemp[i] > maxV)
            {
                maxV = cpuTemp[i];
                std::cout << maxV << std::endl;
            }
        }
        
        tensorFloat64 gpuMin = gpuTemp.reduceMinAll();
        tensorFloat64 gpuMax = gpuTemp.reduceMaxAll();
        std::vector<double> cpuMax = gpuMax.toCpu();
        std::vector<double> cpuMin = gpuMin.toCpu();

        tensorFloat64 gpuRes = gpuTemp.reduceSum({0, 1, 2});

        cpuRes = gpuRes.toCpu();
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
            std::cout << i << "," << cpuRes[i] << std::endl;
            // std::cout  << i << ", [";
            // for(size_t j = 0; j < 4; j++)
            //     std::cout << std::hex << (int)(((unsigned char*)&cpuRes[i])[j]) << ", ";
            // std::cout << "]" << std::endl;
        }
    }
};