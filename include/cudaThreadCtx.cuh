#pragma once

#include "helpers.cuh"

#include <cnmem.h>
#include <cudnn.h>
#include <cuda.h>
#include <memory>

class cudaThreadCtx
{
private:
    /* data */
public:
    cudnnHandle_t cudnn;
    //std::shared_ptr<cnmemDevice_t> device;
    cnmemDevice_t* device;
    cudaStream_t stream;
    cudaThreadCtx()
    {

    }
    ~cudaThreadCtx()
    {
        gpuErrchk( cudnnDestroy(cudnn) );
        gpuErrchk( cudaStreamDestroy(stream) );
    }

    void createCudnnHandle(int device, bool restoreDevice = true)
    {
        int oldDevice;
        if(restoreDevice)
        {
            gpuErrchk( cudaGetDevice(&oldDevice) );
            errChk();
        }

        gpuErrchk( cudaSetDevice(device) );
        errChk();

        gpuErrchk( cudnnCreate(&cudnn) );
        errChk();

        if(restoreDevice)
        {
            gpuErrchk( cudaSetDevice(oldDevice) );
            errChk();
        }
    }

    void createStream(int device, bool restoreDevice = true)
    {
        int oldDevice;
        if(restoreDevice)
        {
            gpuErrchk( cudaGetDevice(&oldDevice) );
            errChk();
        }

        gpuErrchk( cudaSetDevice(device) );
        errChk();

        gpuErrchk( cudaStreamCreate(&stream) );
        errChk();

        if(restoreDevice)
        {
            gpuErrchk( cudaSetDevice(oldDevice) );
            errChk();
        }
    }

    void setDevice()
    {
        gpuErrchk( cudaSetDevice(device->device) );
        errChk();
    }

    void errChk()
    {
        gpuErrchk( cudaPeekAtLastError() );
	    gpuErrchk( cudaDeviceSynchronize() );
    }
};
