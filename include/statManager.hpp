#pragma once

#include <vector>

#include "stat.hpp"
#include "pixelFreq.cuh"
#include "frequency.cuh"
#include "imgFreq.cuh"

class statManager
{
private:
    bool useFreq = false;
    bool usePixelFreq = false;
    bool useImgFreq = false;
public:
    statManager() {}
    ~statManager() {}
    void addPixelFrequency()
    {
        usePixelFreq = true;
    }
    void addFrequency()
    {
        useFreq = true;
    }
    void addImageFrequency()
    {
        useImgFreq = true;
    }
    void addAll()
    {
        useFreq = true;
        usePixelFreq = true;
        useImgFreq = true;
    }
    std::vector<std::shared_ptr<stat>> getStatList(int maxClass)
    {
        std::vector<std::shared_ptr<stat>> ret;
        if(usePixelFreq || useFreq)
        {
            pixelFreq* pixFreq = new pixelFreq(maxClass);
            if(useFreq)
            {
                std::shared_ptr<pixelFreq> temp(pixFreq);
                ret.push_back(std::shared_ptr<stat>(new frequency(temp, usePixelFreq)));
            }
            else
                ret.push_back(std::shared_ptr<stat>(pixFreq));
        }
        if(useImgFreq)
        {
            ret.push_back(std::shared_ptr<stat>(new imgFreq(maxClass)));
        }
        return ret;
    }
};

