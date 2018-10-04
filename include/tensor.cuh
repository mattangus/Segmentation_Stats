#pragma once

#include "helpers.cuh"

#include <cuda.h>
#include <cudnn.h>

template<typename T>
class tensor
{
private:
public:
    T* data;
    int n, h, w, d;
    cudnnHandle_t cudnn;
    tensor(cudnnHandle_t& cudnn, int n, int h, int w, int d) : cudnn(cudnn), n(n), h(h), w(w), d(d)
    {

    }

    tensor(cudnnHandle_t& cudnn, int n, int h, int w, int d, bool alloc) : tensor(cudnn,n,h,w,d)
    {
        if(alloc)
            malloc();
    }

    ~tensor()
    {
        gpuErrchk( cudaFree(data) );
    }

    void malloc()
    {
        gpuErrchk( cudaMalloc((void**) &data, n*h*w*d*sizeof(T)) );
    }
    
    
};
