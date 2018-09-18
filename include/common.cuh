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

        if (x >= w || y >= h) return;

        int ind = (y*w + x)*d;

        gpuC[ind] = gpuA[ind] + gpuB[ind];
    }
}

template <typename T>
void add(T* gpuA, T* gpuB, T* gpuC, int h, int w, int d)
{
    dim3 threads(16,16);
	dim3 blocks((w/threads.x)+1, (h/threads.y)+1); // blocks running on core
    cudaKernels::addOp<<<blocks, threads>>>(gpuA, gpuB, gpuC, h, w, d);
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