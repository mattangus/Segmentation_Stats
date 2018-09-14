#pragma once

#include "helpers.cuh"
#include "stat.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

class frequency : stat<char*>
{
protected:
public:
    frequency(int numThreads) : numThreads(numThreads)
    {

    }
    ~frequency(){ }
    void accumulate(char* d_obj, int thread)
    {

    }
    void finalize()
    {

    }
    void merge()
    {

    }
};