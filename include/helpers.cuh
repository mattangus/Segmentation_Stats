#pragma once

#include <stdlib.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <exception>
#include <sstream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudnnStatus_t code, const char *file, int line, bool abort=true)
{
    if (code != CUDNN_STATUS_SUCCESS) 
    {
        std::stringstream ss;
        ss << "CuDNNassert: (" << code << ") " << cudnnGetErrorString(code) << " " << file << " " << line;
        std::cerr << ss.str() << std::endl;
        if (abort)
        {
            throw std::runtime_error(ss.str());
        }
    }
}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        std::stringstream ss;
        ss << "CUDAassert: (" << code << ") " << cudaGetErrorString(code) << " " << file << " " << line;
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

//taken from tf (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/cuda_device_functions.h)

// Helper for range-based for loop using 'delta' increments.
// Usage: see CudaGridRange?() functions below.
template <typename T>
class CudaGridRange {
  struct Iterator {
    __device__ Iterator(T index, T delta) : index_(index), delta_(delta) {}
    __device__ T operator*() const { return index_; }
    __device__ Iterator& operator++() {
      index_ += delta_;
      return *this;
    }
    __device__ bool operator!=(const Iterator& other) const {
      bool greater = index_ > other.index_;
      bool less = index_ < other.index_;
      // Anything past an end iterator (delta_ == 0) is equal.
      // In range-based for loops, this optimizes to 'return less'.
      if (!other.delta_) {
        return less;
      }
      if (!delta_) {
        return greater;
      }
      return less || greater;
    }

   private:
    T index_;
    const T delta_;
  };

 public:
  __device__ CudaGridRange(T begin, T delta, T end)
      : begin_(begin), delta_(delta), end_(end) {}

  __device__ Iterator begin() const { return Iterator{begin_, delta_}; }
  __device__ Iterator end() const { return Iterator{end_, 0}; }

 private:
  T begin_;
  T delta_;
  T end_;
};

// Helper to visit indices in the range 0 <= i < count, using the x-coordinate
// of the global thread index. That is, each index i is visited by all threads
// with the same x-coordinate.
// Usage: for(int i : CudaGridRangeX(count)) { visit(i); }
template <typename T>
__device__ CudaGridRange<T> CudaGridRangeX(T count) {
  return CudaGridRange<T>(blockIdx.x * blockDim.x + threadIdx.x,
                                  gridDim.x * blockDim.x, count);
}