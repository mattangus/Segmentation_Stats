#pragma once

#include <cuda.h>
#include <cudnn.h>
#include <functional>
#include <thread>

#include "helpers.cuh"
#include "cudaThreadCtx.cuh"

#define IND2(x, y, h, w) ((y)*(w) + (x))
#define IND3(x, y, z, h, w, d) (((y)*(w) + (x))*(d) + (z))
#define IND4(i, x, y, z, n, h, w, d) ((((i)*(n) + (y))*(w) + (x))*(d) + (z))

#define kernalBiOp(name, op)                         \
    template <typename T>                            \
    __global__                                       \
    void name(T* gpuA, T* gpuB, T* gpuC, int count)  \
    {                                                \
        for(int ind : CudaGridRangeX(count))         \
            gpuC[ind] = gpuA[ind] op gpuB[ind];      \
    }                                                \

namespace cudaKernels
{
    // template <typename T> 
    // __global__
    // void addOp(T* gpuA, T* gpuB, T* gpuC, int h, int w, int d)
    // {
    //     const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    //     const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    //     const int z = (blockIdx.z * blockDim.z) + threadIdx.z;

    //     if (x >= w || y >= h || z >= d) return;

    //     int ind = IND3(x,y,z,h,w,d);

    //     gpuC[ind] = gpuA[ind] + gpuB[ind];
    // }

    kernalBiOp(addOp, +);
    kernalBiOp(subOp, -);
    kernalBiOp(divOp, /);
    kernalBiOp(mulOp, *);

    template<typename T1, typename T2>
    __global__
    void castOp(T1* gpuA, T2* gpuB, int count)
    {
        for(int x : CudaGridRangeX(count))
        {
            gpuB[x] = (T2)(gpuA[x]);
        }
    }

    template<typename T>
    __global__
    void oneHotOp(T* gpuA, T* gpuB, int n, int h, int w, int maxClass, T on)
    {
        const int imN = (blockIdx.x * blockDim.x) + threadIdx.x;
        const int x = (blockIdx.y * blockDim.y) + threadIdx.y;
        const int y = (blockIdx.z * blockDim.z) + threadIdx.z;
        // only support d = 1

        if (imN >= n || x >= w || y >= h) return;

        int ind = (imN*n*h + y)*w + x; //no c in input
        unsigned char c = gpuA[ind];
        if(c >= maxClass)
            c = maxClass - 1;
        ind = ((imN*n + c)*h + y)*w + x;
        gpuB[ind] = on;
    }

}

template <typename T>
void add(T* gpuA, T* gpuB, T* gpuC, int numElem, cudaStream_t stream)
{
    int kTPB = 1024;
    LAUNCH(cudaKernels::addOp, numElem, kTPB, 0, stream)(gpuA, gpuB, gpuC, numElem);
    #ifdef SYNC_STREAM
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaStreamSynchronize(stream) );
    #endif
}

template <typename T>
void sub(T* gpuA, T* gpuB, T* gpuC, int numElem, cudaStream_t stream)
{
    int kTPB = 1024;
    LAUNCH(cudaKernels::subOp, numElem, kTPB, 0, stream)(gpuA, gpuB, gpuC, numElem);
    #ifdef SYNC_STREAM
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaStreamSynchronize(stream) );
    #endif
}

template <typename T>
void div(T* gpuA, T* gpuB, T* gpuC, int numElem, cudaStream_t stream)
{
    int kTPB = 1024;
    LAUNCH(cudaKernels::divOp, numElem, kTPB, 0, stream)(gpuA, gpuB, gpuC, numElem);
    #ifdef SYNC_STREAM
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaStreamSynchronize(stream) );
    #endif
}

template <typename T>
void mul(T* gpuA, T* gpuB, T* gpuC, int numElem, cudaStream_t stream)
{
    int kTPB = 1024;
    LAUNCH(cudaKernels::mulOp, numElem, kTPB, 0, stream)(gpuA, gpuB, gpuC, numElem);
    #ifdef SYNC_STREAM
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaStreamSynchronize(stream) );
    #endif
}

template<typename T1, typename T2>
void castWrapper(T1* gpuA, T2* gpuB, int numElem, cudaStream_t stream)
{
    int kTPB = 1024;
    LAUNCH(cudaKernels::castOp, numElem, kTPB, 0, stream)(gpuA, gpuB, numElem);
    #ifdef SYNC_STREAM
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaStreamSynchronize(stream) );
    #endif
}

template <typename T>
void oneHotWrapper(T* gpuA, T* gpuB, int n, int h, int w, int maxClass, T on, cudaStream_t stream)
{
    dim3 blockDim(2,16,16);
	dim3 blocks((n/blockDim.x)+1, (w/blockDim.y)+1, (h/blockDim.z)+1);
    cudaKernels::oneHotOp<<<blocks, blockDim, 0, stream>>>(gpuA, gpuB, n, h, w, maxClass, on);
    #ifdef SYNC_STREAM
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaStreamSynchronize(stream) );
    #endif
}

template<typename T>
cudnnDataType_t getCudnnType()
{
    if(std::is_same<T, float>::value)
        return CUDNN_DATA_FLOAT;
    else if(std::is_same<T, double>::value)
        return CUDNN_DATA_DOUBLE;
    else if(std::is_same<T, int>::value)
        return CUDNN_DATA_INT32;
    else if(std::is_same<T, char>::value)
        return CUDNN_DATA_INT8;
    else
        throw std::runtime_error("Cannot use any other type of");
}

template<typename T>
void _op(cudaThreadCtx* ctx, T* gpuA, T* gpuB, T** gpuC, int h, int w, int d, cudnnOpTensorOp_t opType)
{
    cudnnDataType_t dType = getCudnnType<T>();

    cudnnOpTensorDescriptor_t opDescriptor;
    gpuErrchk( cudnnCreateOpTensorDescriptor(&opDescriptor) );
    gpuErrchk( cudnnSetOpTensorDescriptor(opDescriptor,
                                opType,
                                dType,
                                CUDNN_NOT_PROPAGATE_NAN) );

    

    // cudnnStatus_t cudnnOpTensor(
    // cudnnHandle_t                     handle,
    // const cudnnOpTensorDescriptor_t   opTensorDesc,
    // const void                       *alpha1,
    // const cudnnTensorDescriptor_t     aDesc,
    // const void                       *A,
    // const void                       *alpha2,
    // const cudnnTensorDescriptor_t     bDesc,
    // const void                       *B,
    // const void                       *beta,
    // const cudnnTensorDescriptor_t     cDesc,
    // void                             *C)
    //C = op ( alpha1[0] * A, alpha2[0] * B ) + beta[0] * C
}

void setReduceDims(std::initializer_list<int> axes, int* outN, int* outH, int* outW, int* outD)
{
    for(int ax : axes)
    {
        if(ax == 0)
            *outN = 1;
        else if(ax == 1)
            *outH = 1;
        else if(ax == 2)
            *outW = 1;
        else if(ax == 3)
            *outD = 1;
    }
}

template<typename T>
void _reduce(cudaThreadCtx* ctx, T* gpuA, T** gpuB, std::initializer_list<int> axes, int n, int h, int w, int d, cudnnReduceTensorOp_t reduceType)
{
    int outN = n, outH = h, outW = w, outD = d;
    setReduceDims(axes, &outN, &outH, &outW, &outD);
    

    //gpuErrchk( cudaMalloc(gpuB, outN*outH*outW*outD*sizeof(T)) );
    gpuErrchk( cnmemMalloc((void**) gpuB, outN*outH*outW*outD*sizeof(T), ctx->stream) );
    gpuErrchk( cudaMemset(*gpuB, 0, outN*outH*outW*outD*sizeof(T)) );

    cudnnDataType_t dType = getCudnnType<T>();

    cudnnTensorDescriptor_t inputDescriptor;
    gpuErrchk( cudnnCreateTensorDescriptor(&inputDescriptor) );
    gpuErrchk( cudnnSetTensor4dDescriptor(inputDescriptor,
                                            CUDNN_TENSOR_NCHW,
                                            dType,
                                            n, d, h, w) );

    cudnnTensorDescriptor_t outputDescriptor;
    gpuErrchk( cudnnCreateTensorDescriptor(&outputDescriptor) );
    gpuErrchk( cudnnSetTensor4dDescriptor(outputDescriptor,
                                            CUDNN_TENSOR_NCHW,
                                            dType,
                                            outN, outD, outH, outW) );

    cudnnReduceTensorDescriptor_t reduceTensorDesc;
    gpuErrchk( cudnnCreateReduceTensorDescriptor(&reduceTensorDesc) );
    gpuErrchk( cudnnSetReduceTensorDescriptor(reduceTensorDesc,
                                                reduceType,
                                                dType,
                                                CUDNN_NOT_PROPAGATE_NAN,
                                                CUDNN_REDUCE_TENSOR_NO_INDICES,
                                                CUDNN_8BIT_INDICES) );

    size_t workspaceSize;
    gpuErrchk( cudnnGetReductionWorkspaceSize(ctx->cudnn,
                                                reduceTensorDesc,
                                                inputDescriptor,
                                                outputDescriptor,
                                                &workspaceSize) );

    size_t indicesSize;
    gpuErrchk( cudnnGetReductionIndicesSize(ctx->cudnn,
                                                reduceTensorDesc,
                                                inputDescriptor,
                                                outputDescriptor,
                                                &indicesSize) );

    T alpha = 1;
    T beta = 0;

    void* gpuWorkspace;
    //gpuErrchk( cudaMalloc(&gpuWorkspace, workspaceSize) );
    gpuErrchk( cnmemMalloc(&gpuWorkspace, workspaceSize, ctx->stream) );

    void* gpuIndices;
    //gpuErrchk( cudaMalloc(&gpuIndices, indicesSize) );
    gpuErrchk( cnmemMalloc(&gpuIndices, indicesSize, ctx->stream) );

    gpuErrchk( cudnnReduceTensor(ctx->cudnn,
                                    reduceTensorDesc,
                                    gpuIndices, indicesSize,
                                    gpuWorkspace, workspaceSize,
                                    &alpha,
                                    inputDescriptor, gpuA,
                                    &beta,
                                    outputDescriptor, *gpuB) );
    #ifdef SYNC_STREAM
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaStreamSynchronize(ctx->stream) );
    #endif

    gpuErrchk( cudnnDestroyReduceTensorDescriptor(reduceTensorDesc) );
    gpuErrchk( cudnnDestroyTensorDescriptor(inputDescriptor) );
    gpuErrchk( cudnnDestroyTensorDescriptor(outputDescriptor) );

    gpuErrchk( cnmemFree(gpuIndices, ctx->stream) );
    gpuErrchk( cnmemFree(gpuWorkspace, ctx->stream) );

}

#define reduceFunc(name, type)                                                                          \
template<typename T>                                                                                    \
T* name(cudaThreadCtx* ctx, T* gpuA, std::initializer_list<int> axes, int n, int h, int w, int d)         \
{                                                                                                       \
    T* gpuB;                                                                                            \
    _reduce(ctx, gpuA, &gpuB, axes, n, h, w, d, type);                                                    \
    return gpuB;                                                                                   \
}                                                                                                       \

#define reduceFuncAll(name, type)                           \
template<typename T>                                        \
T* name(cudaThreadCtx* ctx, T* gpuA, int n, int h, int w, int d) \
{                                                           \
    T* gpuRes;                                         \
    _reduce(ctx, gpuA, &gpuRes, {0,1,2,3}, n, h, w, d, type);     \
    return gpuRes;                                          \
}                                                           \


reduceFunc(reduceSumWrapper, CUDNN_REDUCE_TENSOR_ADD);
reduceFunc(reduceMaxWrapper, CUDNN_REDUCE_TENSOR_MAX);
reduceFunc(reduceMinWrapper, CUDNN_REDUCE_TENSOR_MIN);
reduceFunc(reduceMeanWrapper, CUDNN_REDUCE_TENSOR_AVG);
reduceFunc(reduceProdWrapper, CUDNN_REDUCE_TENSOR_MUL);

reduceFuncAll(reduceSumAllWrapper, CUDNN_REDUCE_TENSOR_ADD);
reduceFuncAll(reduceMaxAllWrapper, CUDNN_REDUCE_TENSOR_MAX);
reduceFuncAll(reduceMinAllWrapper, CUDNN_REDUCE_TENSOR_MIN);
reduceFuncAll(reduceMeanAllWrapper, CUDNN_REDUCE_TENSOR_AVG);
reduceFuncAll(reduceProdAllWrapper, CUDNN_REDUCE_TENSOR_MUL);

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
