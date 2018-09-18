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
        ss << "GPUassert: (" << code << ") " << cudaGetErrorString(code) << " " << file << " " << line;
        std::cerr << ss.str() << std::endl;
        if (abort)
        {
            throw std::runtime_error(ss.str());
        }
    }
}

template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

#define LAUNCH(kernel) kernel<<<(numElem + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>