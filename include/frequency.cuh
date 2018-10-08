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
    std::vector<float> cpuRes;
    std::shared_ptr<pixelFreq> pixelFreqs;
    bool usePixelFreq; // if true then pixel freqs should be saved as well
public:
    frequency(std::shared_ptr<pixelFreq>& pixelFreqs, bool usePixelFreq) : pixelFreqs(pixelFreqs), usePixelFreq(usePixelFreq)
    {
        this->name = "frequency";
    }
    ~frequency() { }
    void accumulate(cudnnHandle_t& cudnn, tensorUint8& gpuIm, std::string& path)
    {
        pixelFreqs->accumulate(cudnn, gpuIm, path);
    }
    void finalize(cudnnHandle_t& cudnn)
    {
        //nothing to finialize
    }
    void merge(cudnnHandle_t& cudnn)
    {
        pixelFreqs->merge(cudnn);

        tensor<float> gpuTemp = pixelFreqs->gpuRes->cast<float>();

        // tensor<float> gpuMin = gpuTemp.reduceMinAll();
        // tensor<float> gpuMax = gpuTemp.reduceMaxAll();

        tensor<float> gpuRes = gpuTemp.reduceSum({0, 1, 2});

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
        std::cout << "class,count" << std::endl;
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