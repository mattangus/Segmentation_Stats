#pragma once

#include "helpers.cuh"
#include "common.cuh"
#include "transform.cuh"
#include "cudaThreadCtx.cuh"

#include <cnmem.h>

#include <cuda.h>
#include <cudnn.h>

#define CUDNN_CALL(expr)    \
    NHWCToNCHW();           \
    (expr);                 \
    NCHWToNHWC();           \

template<typename T>
class tensor
{
private:
    // A GPU helper functor that converts NHWC TensorFlow data format to
    // NCHW format that is accepted by Cudnn.
    tensor<T> NHWCToNCHW(bool swapMem = true) {
        Dimension<3> combined_dims;
        combined_dims[0] = n;  // N (batch)
        combined_dims[1] = h * w;  // spatial dimensions (HW)
        combined_dims[2] = d;  // C (channels)

        tensor<T> out(ctx, n, h, w, d, true);

        RunSwapDimension1And2InTensor3(data, combined_dims, out.data, numElements(), ctx->stream);

        if(swapMem)
        {
            //swap pointers and let destructor dealloc
            std::swap(out.data, data);
            // T* temp = out.data;
            // out.data = data;
            // data = temp;
        }
        return out;
    }

    // A GPU helper functor that converts NCHW Cudnn data format to NHWC TensorFlow
    // Format.
    tensor<T> NCHWToNHWC(bool swapMem = true) {
        Dimension<3> combined_dims;
        combined_dims[0] = n;  // N (batch)
        combined_dims[1] = d;  // C (channel)
        combined_dims[2] = h * w;  // spatial dimensions (HW)

        tensor<T> out(ctx, n, h, w, d, true);

        RunSwapDimension1And2InTensor3(data, combined_dims, out.data, numElements(), ctx->stream);

        if(swapMem)
        {
            //swap pointers and let destructor dealloc
            std::swap(out.data, data);
            // T* temp = out.data;
            // out.data = data;
            // data = temp;
        }
        return out;
    }

    void handleFormat(cudnnTensorFormat_t format)
    {
        if(format == CUDNN_TENSOR_NHWC)
            NHWCToNCHW();
        else if(format != CUDNN_TENSOR_NCHW)
            throw std::runtime_error("Tensor format must be either 'NHWC' or 'NCHW'");
    }

    tensor(cudaThreadCtx* ctx, T* data, int n, int h, int w, int d)
        : tensor(ctx, data, CUDNN_TENSOR_NCHW, n, h, w, d)
    {
        
    }
    
    //give access to private functions of other datatypes
    template<class U>
    friend class tensor;

    T* data = nullptr;
public:
    cudaThreadCtx* ctx;
    int n, h, w, d;
    cudnnTensorFormat_t format;
    tensor()
    {

    }

    tensor(cudaThreadCtx* ctx) : ctx(ctx) { }

    tensor(cudaThreadCtx* ctx, int n, int h, int w, int d) : ctx(ctx), n(n), h(h), w(w), d(d)
    {

    }

    tensor(cudaThreadCtx* ctx, T* data, cudnnTensorFormat_t format, int n, int h, int w, int d)
        : ctx(ctx), data(data), n(n), h(h), w(w), d(d)
    {
        handleFormat(format);
    }

    tensor(cudaThreadCtx* ctx, int n, int h, int w, int d, bool alloc) : tensor(ctx,n,h,w,d)
    {
        if(alloc)
            allocateMem();
    }

    tensor(tensor<T>&& other) : tensor(other.ctx, other.data, other.n, other.h, other.w, other.d)
    {
        other.data = nullptr;
    }

    tensor(const tensor<T>&) = delete;

    ~tensor()
    {
        gpuErrchk( cnmemFree(data, ctx->stream) );
    }

    T* getData()
    {
        return data;
    }

    void setDims(int n, int h, int w, int d)
    {
        setDims(n, h, w, d, false);
    }

    void setDims(int n, int h, int w, int d, bool alloc)
    {
        this->n = n;
        this->h = h;
        this->w = w;
        this->d = d;
        if(alloc)
            allocateMem();
    }

    void allocateMem()
    {
        //free data if already allocated
        gpuErrchk( cnmemFree(data, ctx->stream) );
        //gpuErrchk( cudaMalloc((void**) &data, getSize()) );
        // size_t freeb1, freeb2, freea1, freea2;
        // size_t totalb1, totalb2, totala1, totala2;
        // cudaMemGetInfo(&freeb1, &totalb1);
        // cnmemMemGetInfo(&freeb2, &totalb2, ctx->stream);
        gpuErrchk( cnmemMalloc((void**) &data, getSize(), ctx->stream) );
        // cudaMemGetInfo(&freea1, &totala1);
        // cnmemMemGetInfo(&freea2, &totala2, ctx->stream);

    }
    
    int numElements()
    {
        return n*h*w*d;
    }

    int getSize()
    {
        return numElements()*sizeof(T);
    }

    void set(T val)
    {
        gpuErrchk( cudaMemset((void*) data, (int)val, getSize()) );
    }
    
    std::vector<T> toCpu()
    {
        std::vector<T> cpuRes(numElements());
        tensor<T> out = NCHWToNHWC(false); //change format for cpu, and don't swap pointers
        gpuErrchk( cudaMemcpy(&cpuRes[0], out.data, getSize(), cudaMemcpyDeviceToHost) );
        return cpuRes;
    }

    void toGpu(T* cpuData, cudnnTensorFormat_t format)
    {
        gpuErrchk( cudaMemcpy(data, cpuData, getSize(), cudaMemcpyHostToDevice) );
        handleFormat(format);
    }

    void toGpu(std::vector<T> cpuData, cudnnTensorFormat_t format)
    {
        if((size_t)numElements() != cpuData.size())
            throw std::runtime_error("tensor, data size mismatch");
        toGpu(&cpuData[0], format);
    }

    bool isCompatable(int n, int h, int w, int d)
    {
        return (n == this->n && h == this->h && w == this->w && d == this->d);
    }

    tensor<T> operator+(const tensor<T>& other)
    {
        if(n != other.n ||
            h != other.h ||
            w != other.w ||
            d != other.d)
            throw std::runtime_error("Cannot add operations with different dimension");
        tensor<T> ret(ctx, n, h, w, d, true);
        add(data, other.data, ret.data, numElements(), ctx->stream);
        return ret;
    }

    tensor<T>& operator+=(const tensor<T>& other)
    {
        if(n != other.n ||
            h != other.h ||
            w != other.w ||
            d != other.d)
            throw std::runtime_error("Cannot add operations with different dimension");
        add(data, other.data, data, numElements(), ctx->stream);
        return *this;
    }

    tensor<T> reduceSum(std::initializer_list<int> axes)
    {
        int outN = n, outH = h, outW = w, outD = d;
        setReduceDims(axes, &outN, &outH, &outW, &outD);

        T* res = reduceSumWrapper(ctx, data, axes, n, h, w, d);

        return tensor<T>(ctx, res, outN, outH, outW, outD);
    }

    tensor<T> reduceMax(std::initializer_list<int> axes)
    {
        int outN = n, outH = h, outW = w, outD = d;
        setReduceDims(axes, &outN, &outH, &outW, &outD);

        T* res = reduceMaxWrapper(ctx, data, axes, n, h, w, d);

        return tensor<T>(ctx, res, outN, outH, outW, outD);
    }
    
    tensor<T> reduceMin(std::initializer_list<int> axes)
    {
        int outN = n, outH = h, outW = w, outD = d;
        setReduceDims(axes, &outN, &outH, &outW, &outD);

        T* res = reduceMinWrapper(ctx, data, axes, n, h, w, d);

        return tensor<T>(ctx, res, outN, outH, outW, outD);
    }
    
    tensor<T> reduceMean(std::initializer_list<int> axes)
    {
        int outN = n, outH = h, outW = w, outD = d;
        setReduceDims(axes, &outN, &outH, &outW, &outD);

        T* res = reduceMeanWrapper(ctx, data, axes, n, h, w, d);

        return tensor<T>(ctx, res, outN, outH, outW, outD);
    }
    
    tensor<T> reduceProd(std::initializer_list<int> axes)
    {
        int outN = n, outH = h, outW = w, outD = d;
        setReduceDims(axes, &outN, &outH, &outW, &outD);

        T* res = reduceProdWrapper(ctx, data, axes, n, h, w, d);

        return tensor<T>(ctx, res, outN, outH, outW, outD);
    }
    
    tensor<T> reduceSumAll()
    {
        T* res = reduceSumAllWrapper(ctx, data, n, h, w, d);

        return tensor<T>(ctx, res, 1, 1, 1, 1);
    }

    tensor<T> reduceMaxAll()
    {
        T* res = reduceMaxAllWrapper(ctx, data, n, h, w, d);

        return tensor<T>(ctx, res, 1, 1, 1, 1);
    }
    
    tensor<T> reduceMinAll()
    {
        T* res = reduceMinAllWrapper(ctx, data, n, h, w, d);

        return tensor<T>(ctx, res, 1, 1, 1, 1);
    }
    
    tensor<T> reduceMeanAll()
    {
        T* res = reduceMeanAllWrapper(ctx, data, n, h, w, d);

        return tensor<T>(ctx, res, 1, 1, 1, 1);
    }
    
    tensor<T> reduceProdAll()
    {
        T* res = reduceProdAllWrapper(ctx, data, n, h, w, d);

        return tensor<T>(ctx, res, 1, 1, 1, 1);
    }
    
    template<typename U> operator tensor<U>()
    {
        //cast(T1* gpuA, T2* gpuB, int numElem)
        tensor<U> ret(ctx, n, h, w, d, true);
        cast(data, ret.data, numElements(), ctx->stream);
        return ret;
    }
    
    template<typename U>
    tensor<U> cast()
    {
        tensor<U> ret(ctx, n, h, w, d, true);
        castWrapper(data, ret.data, numElements(), ctx->stream);
        return ret;
    }

    tensor<T> oneHot(int numClass, T on = 1, T off = 0)
    {
        if(d != 1)
            throw std::runtime_error("5D tensor not implemented");

        tensor<T> ret(ctx, n, h, w, numClass, true);
        ret.set(off);
        oneHotWrapper(data, ret.data, n, h, w, numClass, on, ctx->stream);
        return ret;        
    }
};
