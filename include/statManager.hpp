#pragma once

#include <vector>

#include "stat.hpp"
#include "pixelFreq.cuh"
#include "frequency.cuh"

class statManager
{
private:
    bool useFreq;
    bool usePixelFreq;
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
    std::vector<std::shared_ptr<stat>> getStatList()
    {
        std::vector<std::shared_ptr<stat>> ret;
        if(usePixelFreq || useFreq)
        {
            pixelFreq* pixFreq = new pixelFreq();
            if(useFreq)
            {
                std::shared_ptr<pixelFreq> temp(pixFreq);
                ret.push_back(std::shared_ptr<stat>(new frequency(temp, usePixelFreq)));
            }
            else
                ret.push_back(std::shared_ptr<stat>(pixFreq));
        }
        return ret;
    }
};

