#pragma once

#include <vector>
#include <cudnn.h>
#include <cuda.h>
#include <string>

#include "tensor.cuh"
#include "testHelper.cuh"

#define PRINT_SHAPE(t) std::cout << #t << ".shape: [" << (t).n << "," << (t).h << "," << (t).w << "," << (t).d << "]" << std::endl

namespace tensorTests
{

    void setup()
    {

    }

    void testReduceSum1D()
    {
        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};
            tensor<float> gpuIn(cudnn, 1, 1, 1, 10, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({3});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {45.0f}, "Reduce Sum 1d Depth");
        }

        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};
            tensor<float> gpuIn(cudnn, 1, 1, 10, 1, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({2});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {45.0f}, "Reduce Sum 1d Width");
        }

        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};
            tensor<float> gpuIn(cudnn, 1, 10, 1, 1, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({1});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {45.0f}, "Reduce Sum 1d Height");
        }

        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};
            tensor<float> gpuIn(cudnn, 10, 1, 1, 1, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({0});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {45.0f}, "Reduce Sum 1d Number");
        }
    }

    void testReduceSum2D()
    {
        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};
            tensor<float> gpuIn(cudnn, 1, 1, 2, 5, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({3});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {10.0f, 35.0f}, "Reduce Sum 2d Depth");
        }

        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};
            tensor<float> gpuIn(cudnn, 1, 1, 5, 2, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({2});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {20.0f, 25.0f}, "Reduce Sum 2d Depth");
        }

        {
            std::vector<float> in = {3,5,7,11,13,17,19,23,29,31};
            tensor<float> gpuIn(cudnn, 1, 1, 5, 2, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({3});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {8.0f,18.0f,30.0f,42.0f,60.0f}, "Reduce Sum 2d Long");
        }

        {
            std::vector<float> in = {3,5,7,11,13,17,19,23,29,31};
            tensor<float> gpuIn(cudnn, 1, 5, 1, 2, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({3});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {8.0f,18.0f,30.0f,42.0f,60.0f}, "Reduce Sum 2d Long");
        }

        {
            std::vector<float> in = {3,5,7,11,13,17,19,23,29,31};
            tensor<float> gpuIn(cudnn, 5, 1, 1, 2, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({3});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {8.0f,18.0f,30.0f,42.0f,60.0f}, "Reduce Sum 2d Long");
        }


        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};
            tensor<float> gpuIn(cudnn, 1, 2, 5, 1, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({2});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {10.0f, 35.0f}, "Reduce Sum 2d Width");
        }

        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};
            tensor<float> gpuIn(cudnn, 1, 5, 2, 1, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({1});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {20.0f, 25.0f}, "Reduce Sum 2d Height");
        }

        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};
            tensor<float> gpuIn(cudnn, 1, 2, 5, 1, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({1});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {5.0f,7.0f,9.0f,11.0f,13.0f}, "Reduce Sum 2d Long");
        }

        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};
            tensor<float> gpuIn(cudnn, 5, 2, 1, 1, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({0});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {20.0f, 25.0f}, "Reduce Sum 2d Number");
        }
    }
    
    void runAllTests()
    {
        testReduceSum1D();
        testReduceSum2D();
    }
};
