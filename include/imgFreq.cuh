#pragma once
#include <opencv2/opencv.hpp>
#include <cudnn.h>
#include <mutex>

#include "tensor.cuh"
#include "types.cuh"
#include "stat.hpp"
/**
 * @brief 
 *
 */
class imgFreq : public stat
{
protected:
    std::string name;
    thread_map<std::unordered_map<std::string, std::vector<double>>> freqMap;
    std::unordered_map<std::string, std::vector<double>> finalResult;
    const int maxClass;
public:
    imgFreq(int maxClass) : name("ImageFreq"), maxClass(maxClass) { }
    ~imgFreq() { }
    void accumulate(cudnnHandle_t& cudnn, tensorUint8& gpuObj, std::string& path)
    {
        tensor<double> temp = gpuObj.cast<double>();
        std::vector<double> result = temp.oneHot(maxClass).reduceSum({0, 1, 2}).toCpu();

        if(!freqMap.hasData())
        {
            std::unordered_map<std::string, std::vector<double>> v;
            freqMap.set(v);
        }

        freqMap.get()[path] = result;
    }

    void finalize(cudnnHandle_t& cudnn)
    {
        //nothing to finalize
    }
    void merge(cudnnHandle_t& cudnn)
    {
        for (auto & it : freqMap.toList()) {
            //add(gpuRes->data, it.second->data, gpuRes->data, h, w, maxClass);
            for(auto& item : it)
            {
                finalResult[item.first] = item.second;
            }
        }
    }
    void save(std::string outputFolder)
    {
        fs::path base = outputFolder;
        fs::path outFile = base / (name + ".csv");

        std::ofstream csv(outFile);

        csv << "path,";
        for(int i = 0; i < maxClass; i++)
        {
            csv << i;
            if(i != maxClass -1)
                csv << ",";
        }
        csv << std::endl;

        for(auto& item : finalResult)
        {
            csv << item.first << ",";
            for(int i = 0; i < maxClass; i++)
            {
                csv << item.second[i];
                if(i != maxClass -1)
                    csv << ",";
            }
            csv << std::endl;
        }
    }
};