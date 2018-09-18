#pragma once

#include <cuda.h>
#include <functional>

namespace cudaKernels
{
    template <typename T> 
    __global__
    void addOp(T* gpuA, T* gpuB, T* gpuC, int h, int w, int d)
    {
        const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
        const int z = (blockIdx.z * blockDim.z) + threadIdx.z;

        if (x >= w || y >= h || z >= d) return;

        int ind = (y*w + x)*d + z;

        gpuC[ind] = gpuA[ind] + gpuB[ind];
    }
}

template <typename T>
void add(T* gpuA, T* gpuB, T* gpuC, int h, int w, int d)
{
    dim3 blockDim(8,8,8);
	dim3 blocks((w/blockDim.x)+1, (h/blockDim.y)+1, (d/blockDim.z)+1); // blocks running on core
    cudaKernels::addOp<<<blocks, blockDim>>>(gpuA, gpuB, gpuC, h, w, d);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

/*#define makeBiOp(name, op)   \
    template <typename T>    \
    __device__               \
    T name(T gpuA, T gpuB)   \
    {                        \
        return gpuA op gpuB; \
    }                        \

makeBiOp(addOp, +);
makeBiOp(subOp, -);
makeBiOp(divOp, /);
makeBiOp(mulOp, *);


template <typename T> 
//__global__
void binaryOp(T* gpuA, T* gpuB, T* gpuC, int h, int w, int d, std::function<T(T,T)> op)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= w || y >= h) return;

    int ind = (y*w + x)*d;

    gpuC[ind] = op(gpuA[ind], gpuB[ind]);
}

template <typename T>
void add(T* gpuA, T* gpuB, T* gpuC, int h, int w, int d)
{
    binaryOp(gpuA, gpuB, gpuC, h, w, d, addOp);
}

template <typename T>
void sub(T* gpuA, T* gpuB, T* gpuC, int h, int w, int d)
{
    binaryOp(gpuA, gpuB, gpuC, h, w, d, subOp);
}

template <typename T>
void div(T* gpuA, T* gpuB, T* gpuC, int h, int w, int d)
{
    binaryOp(gpuA, gpuB, gpuC, h, w, d, divOp);
}

template <typename T>
void mul(T* gpuA, T* gpuB, T* gpuC, int h, int w, int d)
{
    binaryOp(gpuA, gpuB, gpuC, h, w, d, mulOp);
}*/