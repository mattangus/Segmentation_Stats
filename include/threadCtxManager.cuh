#pragma once

#include "cudaThreadCtx.cuh"
#include "helpers.cuh"

#include <unordered_map>

// works!!
class threadCtxManager
{
private:
    cudaThreadCtx* ctx;
    cnmemDevice_t* device;
    std::vector<cudaStream_t> streams;
    size_t align(size_t num, size_t multiple)
    {
        if(multiple == 0)
            return num;
        size_t rem = num % multiple;
        if(rem == 0)
            return num;
        
        return num + multiple - rem;
    }
public:
    
    threadCtxManager(int numThread, std::vector<int>& availDevice)
    {
        size_t freev, totalv;
        gpuErrchk( cudaMemGetInfo(&freev, &totalv) );

        size_t toUse = freev * 0.95f;
        size_t perStream = align(toUse / numThread, 8);

        ctx = new cudaThreadCtx[numThread];
        std::vector<size_t> sizes;
        for(int i = 0; i < numThread; i++)
        {
            ctx[i].createCudnnHandle(availDevice[0], false);
            ctx[i].createStream(availDevice[0], false);
            streams.push_back(ctx[i].stream);
            sizes.push_back(perStream);
            gpuErrchk( cudnnSetStream(ctx[i].cudnn, ctx[i].stream) );
        }

        device = new cnmemDevice_t();
        memset(device, 0, sizeof(device));
        device->device = availDevice[0];
        device->size = toUse;
        device->numStreams = numThread;
        device->streams = &streams[0];
        device->streamSizes = &sizes[0];
        gpuErrchk( cnmemInit(1, device, CNMEM_FLAGS_DEFAULT) );

        for(int i = 0; i < numThread; i++)
        {
            ctx[i].device = device;
        }
    }
    ~threadCtxManager()
    {
        if(device)
            delete device;
        if(ctx)
            delete [] ctx;
        cnmemFinalize();
    }
    cudaThreadCtx* operator [](int i)
    {
        return &ctx[i];
    }
};


// class threadCtxManager
// {
// private:
//     int numThread;
//     std::vector<int> availDevice;
//     std::unordered_map<int, std::vector<cudaThreadCtx*>> contexts;
//     std::unordered_map<int, std::vector<size_t>> streamSizes;
//     std::unordered_map<int, std::vector<cudaStream_t> > allStreams;
//     std::vector<std::shared_ptr<cnmemDevice_t>> devices;
//     void createAllContexts()
//     {
//         for(int i = 0; i < numThread; i++)
//         {
//             int d = availDevice[i % availDevice.size()];
            
//             cudaThreadCtx* ctx = new cudaThreadCtx();
//             //ctx->createStream(d, false);
//             ctx->stream = NULL;
//             ctx->createCudnnHandle(d, false);

//             if(allStreams.count(d) == 0)
//             {
//                 std::vector<cudaStream_t> temp;
//                 allStreams[d] = temp;
//             }
//             allStreams[d].push_back(ctx->stream);
            
//             if(contexts.count(d) == 0)
//             {
//                 std::vector<cudaThreadCtx*> temp;
//                 contexts[d] = temp;
//             }
//             contexts[d].push_back(cudaThreadCtx*(ctx));

//             if(streamSizes.count(d) == 0)
//             {
//                 std::vector<size_t> temp;
//                 streamSizes[d] = temp;
//             }
//             streamSizes[d].push_back(1024L*1024L*1024L*3L);
//         }
//     }
//     void createAllDevices()
//     {
//         for(size_t i = 0; i < availDevice.size(); i++)
//         {
//             int d = availDevice[i];

//             gpuErrchk( cudaSetDevice(d) );
//             gpuErrchk( cudaPeekAtLastError() );
//             gpuErrchk( cudaStreamSynchronize(stream) );
            
//             //cudaThreadCtx* ctx = contexts[d][i];

//             std::shared_ptr<cnmemDevice_t> device = std::shared_ptr<cnmemDevice_t>(new cnmemDevice_t());
//             //memset(device, 0, sizeof(device));
//             device->device = d;
//             device->size = 1024L*1024L*1024L*10L; //5GB, should grow?
//             device->numStreams = 0; //allStreams[d].size();
//             //device->streamSizes = &(streamSizes[d][0]);
//             device->streams = NULL; //&(allStreams[d][0]);
//             gpuErrchk( cnmemInit(1, device.get(), CNMEM_FLAGS_DEFAULT) );
//             devices.push_back(device);
//             for(auto& ctx : contexts[d])
//             {
//                 ctx->device = device;
//                 if(ctx->stream != NULL)
//                     gpuErrchk( cnmemRegisterStream(ctx->stream) );
//             }
//         }
//     }
// public:
//     threadCtxManager(int numThread, std::vector<int>& availDevice) : numThread(numThread), availDevice(availDevice)
//     {
//         int oldDevice;
//         gpuErrchk( cudaGetDevice(&oldDevice) );
//         gpuErrchk( cudaPeekAtLastError() );
// 	    gpuErrchk( cudaStreamSynchronize(stream) );

//         createAllContexts();
//         createAllDevices();

//         gpuErrchk( cudaSetDevice(oldDevice) );
//         gpuErrchk( cudaPeekAtLastError() );
// 	    gpuErrchk( cudaStreamSynchronize(stream) );
//     }
//     ~threadCtxManager()
//     {

//     }
//     cudaThreadCtx* operator [](int i)
//     {
//         int d = availDevice[i % availDevice.size()];
//         return contexts[d][i / availDevice.size()];
//     }
// };