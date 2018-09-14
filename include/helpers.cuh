#pragma once

#include <stdlib.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <exception>
#include <sstream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        std::stringstream ss;
        ss << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line;
        std::cerr << ss.str() << std::endl;
        if (abort)
        {
            throw std::exception(ss.str());
        }
    }
}
