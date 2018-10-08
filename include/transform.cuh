#pragma once

#include <vector>

#include "tensor.cuh"

// // A helper function that converts a flat array index into a tensor index.
// template <int IndexCount>
// __host__ __device__ std::vector<int> FlatToTensorIndex(int index, const std::vector<int>& dims)
// {
//     std::vector<int> tensor_index(IndexCount);
//     for (int i = IndexCount - 1; i >= 0; i--) {
//         int new_index = index / dims[i];
//         tensor_index[i] = index - dims[i] * new_index;
//         index = new_index;
//     }
//     return tensor_index;
// }

// // A helper function that converts a tensor index into a flat array index.
// template <int IndexCount>
// __host__ __device__ int TensorIndexToFlat(const std::vector<int>& index, const std::vector<int>& dims) {
//     int flat_index = index[0];
//     for (int i = 1; i < IndexCount; i++) {
//         flat_index = flat_index * dims[i] + index[i];
//     }
//     return flat_index;
// }

// TODO(mjanusz): Move this to a shared util file.
// A simple array that contains data that can be passed between CPU and GPU.
template <typename T, int IndexCount, T DefaultValue>
struct Array {
    __host__ __device__ inline const T& operator[](int index) const {
        return data[index];
    }
    __host__ __device__ inline T& operator[](int index) {
        return data[index];
    }
    __host__ __device__ inline Array() {
        for (int i = 0; i < IndexCount; i++) {
            data[i] = DefaultValue;
        }
    }
    __host__ __device__ inline Array(T a0) {
        data[0] = a0;
        for (int i = 1; i < IndexCount; i++) {
            data[i] = DefaultValue;
        }
    }
    __host__ __device__ inline Array(T a0, T a1) {
        data[0] = a0;
        data[1] = a1;
        for (int i = 2; i < IndexCount; i++) {
            data[i] = DefaultValue;
        }
    }
    __host__ __device__ inline Array(T a0, T a1, T a2) {
        data[0] = a0;
        data[1] = a1;
        data[2] = a2;
        for (int i = 3; i < IndexCount; i++) {
            data[i] = DefaultValue;
        }
    }
    inline Array(const std::array<T, IndexCount>& array) {
        for (int i = 0; i < IndexCount; i++) {
            data[i] = array[i];
        }
    }
    T data[IndexCount];
};

// A dimension type with compile-time known size.
template <int IndexCount>
struct Dimension : Array<int, IndexCount, 1> {
    typedef Array<int, IndexCount, 1> Base;
    __host__ __device__ inline Dimension() : Base() {}
    __host__ __device__ inline Dimension(int a0) : Base(a0) {}
    __host__ __device__ inline Dimension(int a0, int a1)
        : Base(a0, a1) {}
    __host__ __device__ inline Dimension(int a0, int a1, int a2)
        : Base(a0, a1, a2) {}
    inline Dimension(const std::array<int, IndexCount>& array)
        : Base(array) {}
};

// An index type with compile-time known size.
template <int IndexCount>
struct Index : Array<int, IndexCount, 0> {
    typedef Array<int, IndexCount, 0> Base;
    __host__ __device__ inline Index() : Base() {}
    __host__ __device__ inline Index(int a0) : Base(a0) {}
    __host__ __device__ inline Index(int a0, int a1) : Base(a0, a1) {}
    __host__ __device__ inline Index(int a0, int a1, int a2)
        : Base(a0, a1, a2) {}
};

// A helper function that converts a tensor index into a flat array index.
template <int IndexCount>
__host__ __device__ inline int TensorIndexToFlat(const Index<IndexCount>& index, const Dimension<IndexCount>& dims) {
    int flat_index = index[0];
    for (int i = 1; i < IndexCount; i++) {
        flat_index = flat_index * dims[i] + index[i];
    }
    return flat_index;
}

// A helper function that converts a flat array index into a tensor index.
template <int IndexCount>
__host__ __device__ inline Index<IndexCount> FlatToTensorIndex(int index, const Dimension<IndexCount>& dims) {
    Index<IndexCount> tensor_index;
    for (int i = IndexCount - 1; i >= 0; i--) {
        int new_index = index / dims[i];
        tensor_index[i] = index - dims[i] * new_index;
        index = new_index;
    }
    return tensor_index;
}

namespace cudaKernels {
    // // A simple CUDA custom kernel to shuffle dimensions of a 3D tensor according to
    // // the given shuffle permutation in template parameters. Shuffle permutation
    // // <sp0, sp1, sp2> shuffles dimensions such that input dimension 0 goes to sp0,
    // // 1 goes to sp1 and 2 goes to sp2. For example, shuffle permutation <2, 0, 1>
    // // will populate output so that input[x][y][z] is equal to (*output)[y][z][x].
    // //
    // // Requires that nthreads is equal to the total number of elements in the input
    // // tensor.
    // template <typename T, int sp0, int sp1, int sp2>
    // __global__ void ShuffleInTensor3Simple(int nthreads, const T* input, Index<3> input_dims, T* output) {
    //     Index<3> output_dims;
    //     output_dims[sp0] = input_dims[0];
    //     output_dims[sp1] = input_dims[1];
    //     output_dims[sp2] = input_dims[2];

    //     // Iterate over output as opposed to iterating over input for better
    //     // performance. Iterating over output will generate sequential writes and
    //     // random reads that performs better compared to sequential reads and random
    //     // writes.
    //     for(int output_index : CudaGridRangeX(nthreads)) {
    //         Index<3> output_tensor_index = FlatToTensorIndex<IndexCount>(output_index, output_dims);

    //         Index<3> input_tensor_index(IndexCount);
    //         input_tensor_index[0] = output_tensor_index[sp0];
    //         input_tensor_index[1] = output_tensor_index[sp1];
    //         input_tensor_index[2] = output_tensor_index[sp2];

    //         int input_index = TensorIndexToFlat<IndexCount>(input_tensor_index, input_dims);

    //         output[output_index] = input[input_index];
    //     }
    // }
    // A simple CUDA custom kernel to shuffle dimensions of a 3D tensor according to
    // the given shuffle permutation in template parameters. Shuffle permutation
    // <sp0, sp1, sp2> shuffles dimensions such that input dimension 0 goes to sp0,
    // 1 goes to sp1 and 2 goes to sp2. For example, shuffle permutation <2, 0, 1>
    // will populate output so that input[x][y][z] is equal to (*output)[y][z][x].
    //
    // Requires that nthreads is equal to the total number of elements in the input
    // tensor.
    template <typename T, int sp0, int sp1, int sp2>
    __global__ void ShuffleInTensor3Simple(int nthreads, const T* input, Dimension<3> input_dims, T* output) {
        Dimension<3> output_dims;
        output_dims[sp0] = input_dims[0];
        output_dims[sp1] = input_dims[1];
        output_dims[sp2] = input_dims[2];

        // Iterate over output as opposed to iterating over input for better
        // performance. Iterating over output will generate sequential writes and
        // random reads that performs better compared to sequential reads and random
        // writes.
        for(int output_index : CudaGridRangeX(nthreads)) {
            Index<3> output_tensor_index = FlatToTensorIndex(output_index, output_dims);

            Index<3> input_tensor_index;
            input_tensor_index[0] = output_tensor_index[sp0];
            input_tensor_index[1] = output_tensor_index[sp1];
            input_tensor_index[2] = output_tensor_index[sp2];

            int input_index = TensorIndexToFlat(input_tensor_index, input_dims);

            output[output_index] = input[input_index];
        }
    }
};

// Launch the GPU kernel that would swap dimension-1 and dimension-2 in a
// 3D tensor. It looks at the shape of the incoming data, and decides the best
// strategy to launch.
template <typename T>
void RunSwapDimension1And2InTensor3(const T* input, const Dimension<3>& input_dims, T* output, int numElem) {
    
    int kThreadsPerBlock = 1024;
    LAUNCH((cudaKernels::ShuffleInTensor3Simple<T, 0, 2, 1>))(numElem, input, input_dims, output);
    //<<<(numElem + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}