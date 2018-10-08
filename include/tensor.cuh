#pragma once

#include "helpers.cuh"
#include "common.cuh"
#include "transform.cuh"

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

        tensor<T> out(cudnn, n, h, w, d, true);

        RunSwapDimension1And2InTensor3(data, combined_dims, out.data, numElements());

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

        tensor<T> out(cudnn, n, h, w, d, true);

        RunSwapDimension1And2InTensor3(data, combined_dims, out.data, numElements());

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

    tensor(cudnnHandle_t& cudnn, T* data, int n, int h, int w, int d)
        : tensor(cudnn, data, CUDNN_TENSOR_NCHW, n, h, w, d)
    {
        
    }
    
    //give access to private functions of other datatypes
    template<class U>
    friend class tensor;

    T* data = nullptr;
public:
    cudnnHandle_t cudnn;
    int n, h, w, d;

    tensor()
    {

    }

    tensor(cudnnHandle_t& cudnn) : cudnn(cudnn) { }

    tensor(cudnnHandle_t& cudnn, int n, int h, int w, int d) : cudnn(cudnn), n(n), h(h), w(w), d(d)
    {

    }

    tensor(cudnnHandle_t& cudnn, T* data, cudnnTensorFormat_t format, int n, int h, int w, int d)
        : cudnn(cudnn), data(data), n(n), h(h), w(w), d(d)
    {
        handleFormat(format);
    }

    tensor(cudnnHandle_t& cudnn, int n, int h, int w, int d, bool alloc) : tensor(cudnn,n,h,w,d)
    {
        if(alloc)
            allocateMem();
    }

    tensor(tensor<T>&& other) : tensor(other.cudnn, other.data, other.n, other.h, other.w, other.d)
    {
        other.data = nullptr;
    }

    tensor(const tensor<T>&) = delete;

    ~tensor()
    {
        gpuErrchk( cudaFree(data) );
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
        gpuErrchk( cudaFree(data) );
        gpuErrchk( cudaMalloc((void**) &data, getSize()) );
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
        gpuErrchk( cudaMemset(data, val, getSize()) );
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
        tensor<T> ret(cudnn, n, h, w, d, true);
        add(data, other.data, ret.data, numElements());
        return ret;
    }

    tensor<T>& operator+=(const tensor<T>& other)
    {
        if(n != other.n ||
            h != other.h ||
            w != other.w ||
            d != other.d)
            throw std::runtime_error("Cannot add operations with different dimension");
        add(data, other.data, data, numElements());
        return *this;
    }

    tensor<T> reduceSum(std::initializer_list<int> axes)
    {
        int outN = n, outH = h, outW = w, outD = d;
        setReduceDims(axes, &outN, &outH, &outW, &outD);

        T* res = reduceSumWrapper(cudnn, data, axes, n, h, w, d);

        return tensor<T>(cudnn, res, outN, outH, outW, outD);
    }

    tensor<T> reduceMax(std::initializer_list<int> axes)
    {
        int outN = n, outH = h, outW = w, outD = d;
        setReduceDims(axes, &outN, &outH, &outW, &outD);

        T* res = reduceMaxWrapper(cudnn, data, axes, n, h, w, d);

        return tensor<T>(cudnn, res, outN, outH, outW, outD);
    }
    
    tensor<T> reduceMin(std::initializer_list<int> axes)
    {
        int outN = n, outH = h, outW = w, outD = d;
        setReduceDims(axes, &outN, &outH, &outW, &outD);

        T* res = reduceMinWrapper(cudnn, data, axes, n, h, w, d);

        return tensor<T>(cudnn, res, outN, outH, outW, outD);
    }
    
    tensor<T> reduceMean(std::initializer_list<int> axes)
    {
        int outN = n, outH = h, outW = w, outD = d;
        setReduceDims(axes, &outN, &outH, &outW, &outD);

        T* res = reduceMeanWrapper(cudnn, data, axes, n, h, w, d);

        return tensor<T>(cudnn, res, outN, outH, outW, outD);
    }
    
    tensor<T> reduceProd(std::initializer_list<int> axes)
    {
        int outN = n, outH = h, outW = w, outD = d;
        setReduceDims(axes, &outN, &outH, &outW, &outD);

        T* res = reduceProdWrapper(cudnn, data, axes, n, h, w, d);

        return tensor<T>(cudnn, res, outN, outH, outW, outD);
    }
    
    tensor<T> reduceSumAll()
    {
        T* res = reduceSumAllWrapper(cudnn, data, n, h, w, d);

        return tensor<T>(cudnn, res, 1, 1, 1, 1);
    }

    tensor<T> reduceMaxAll()
    {
        T* res = reduceMaxAllWrapper(cudnn, data, n, h, w, d);

        return tensor<T>(cudnn, res, 1, 1, 1, 1);
    }
    
    tensor<T> reduceMinAll()
    {
        T* res = reduceMinAllWrapper(cudnn, data, n, h, w, d);

        return tensor<T>(cudnn, res, 1, 1, 1, 1);
    }
    
    tensor<T> reduceMeanAll()
    {
        T* res = reduceMeanAllWrapper(cudnn, data, n, h, w, d);

        return tensor<T>(cudnn, res, 1, 1, 1, 1);
    }
    
    tensor<T> reduceProdAll()
    {
        T* res = reduceProdAllWrapper(cudnn, data, n, h, w, d);

        return tensor<T>(cudnn, res, 1, 1, 1, 1);
    }
    
    template<typename U> operator tensor<U>()
    {
        //cast(T1* gpuA, T2* gpuB, int numElem)
        tensor<U> ret(cudnn, n, h, w, d, true);
        cast(data, ret.data, numElements());
        return ret;
    }
    
    template<typename U>
    tensor<U> cast()
    {
        tensor<U> ret(cudnn, n, h, w, d, true);
        castWrapper(data, ret.data, numElements());
        return ret;
    }
};
