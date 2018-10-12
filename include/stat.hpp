#pragma once
#include <opencv2/opencv.hpp>
#include <cudnn.h>

#include "tensor.cuh"
#include "types.cuh"
/**
 * @brief Abstract class for tracking statistics
 *
 */
class stat
{
protected:
    std::string name;
public:
    virtual ~stat() { }
    virtual void accumulate(cudaThreadCtx* ctx, tensorUint8& gpuObj, std::string& path) = 0;
    // void accumulate(unsigned char* d_obj, int h, int w, int d)
    // {
    //     accumulate(d_obj, int h, int w, int d, 0);
    // }
    virtual void finalize(cudaThreadCtx* ctx) = 0;
    virtual void merge(cudaThreadCtx* ctx) = 0;
    virtual void save(std::string outputFolder) = 0;
};