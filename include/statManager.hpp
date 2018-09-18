#pragma once

#include <vector>

#include "stat.hpp"
#include "frequency.cuh"

class statManager
{
private:
    bool useFreq;
public:
    statManager() {}
    ~statManager() {}
    void addFrequency()
    {
        useFreq = true;
    }

    std::vector<std::shared_ptr<stat>> getStatList()
    {
        std::vector<std::shared_ptr<stat>> ret;
        if(useFreq)
            ret.push_back(std::shared_ptr<stat>(new frequency()));
        return ret;
    }
};

