#pragma once

#include <vector>
#include <cudnn.h>
#include <cuda.h>
#include <string>

#include "tensor.cuh"
#include "testHelper.cuh"
#include "cudaThreadCtx.cuh"

#define PRINT_SHAPE(t) std::cout << #t << ".shape: [" << (t).n << "," << (t).h << "," << (t).w << "," << (t).d << "]" << std::endl

namespace tensorTests
{

    void setup()
    {

    }

    void testAlloc()
    {
        try
        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};
            tensor<float> gpuIn(ctx, 1, 1, 1, 10, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);
            std::cout << "Alloc Passed" << std::endl;
        }
        catch(std::exception& ex)
        {
            std::cout << "Alloc Fialed: " << ex.what() << std::endl;
        }
    }

    void testAdd()
    {
        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};

            float* gpuA;
            float* gpuB;
            float* gpuC;
            int size = in.size()*sizeof(float);
            gpuErrchk( cnmemMalloc((void**)&gpuA, size, ctx->stream) );
            gpuErrchk( cnmemMalloc((void**)&gpuB, size, ctx->stream) );
            gpuErrchk( cnmemMalloc((void**)&gpuC, size, ctx->stream) );

            gpuErrchk( cudaMemsetAsync(gpuA, 0, size, ctx->stream) );
            gpuErrchk( cudaMemsetAsync(gpuB, 0, size, ctx->stream) );
            gpuErrchk( cudaMemsetAsync(gpuC, 0, size, ctx->stream) );

            gpuErrchk( cudaMemcpyAsync(gpuA, &in[0], size, cudaMemcpyHostToDevice, ctx->stream) );
            gpuErrchk( cudaMemcpyAsync(gpuB, &in[0], size, cudaMemcpyHostToDevice, ctx->stream) );
            gpuErrchk( cudaMemcpyAsync(gpuC, &in[0], size, cudaMemcpyHostToDevice, ctx->stream) );

            int kThreadsPerBlock = 1024;
            cudaKernels::addOp<<<(in.size() + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, ctx->stream>>>(gpuA, gpuB, gpuC, in.size());

            std::vector<float> out(in.size());
            gpuErrchk( cudaMemcpyAsync(&out[0], gpuC, size, cudaMemcpyDeviceToHost, ctx->stream) );
            gpuErrchk( cudaStreamSynchronize (ctx->stream) );

            expect(out, {0.0f,2.0f,4.0f,6.0f,8.0f,10.0f,12.0f,14.0f,16.0f,18.0f}, "Add primitive");
        }

        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};
            tensor<float> gpuInA(ctx, 1, 1, 1, 10, true);
            tensor<float> gpuInB(ctx, 1, 1, 1, 10, true);
            gpuInA.toGpu(in, CUDNN_TENSOR_NHWC);
            gpuInB.toGpu(in, CUDNN_TENSOR_NHWC);

            tensor<float> res = gpuInA + gpuInB;
            std::vector<float> out = res.toCpu();

            expect(out, {0.0f,2.0f,4.0f,6.0f,8.0f,10.0f,12.0f,14.0f,16.0f,18.0f}, "Add");
        }
    }

    void testReduceSum1D()
    {
        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};
            tensor<float> gpuIn(ctx, 1, 1, 1, 10, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({3});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {45.0f}, "Reduce Sum 1d Depth");
        }

        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};
            tensor<float> gpuIn(ctx, 1, 1, 10, 1, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({2});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {45.0f}, "Reduce Sum 1d Width");
        }

        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};
            tensor<float> gpuIn(ctx, 1, 10, 1, 1, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({1});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {45.0f}, "Reduce Sum 1d Height");
        }

        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};
            tensor<float> gpuIn(ctx, 10, 1, 1, 1, true);
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
            tensor<float> gpuIn(ctx, 1, 1, 2, 5, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({3});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {10.0f, 35.0f}, "Reduce Sum 2d Depth");
        }

        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};
            tensor<float> gpuIn(ctx, 1, 1, 5, 2, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({2});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {20.0f, 25.0f}, "Reduce Sum 2d Depth");
        }

        {
            std::vector<float> in = {3,5,7,11,13,17,19,23,29,31};
            tensor<float> gpuIn(ctx, 1, 1, 5, 2, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({3});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {8.0f,18.0f,30.0f,42.0f,60.0f}, "Reduce Sum 2d Long");
        }

        {
            std::vector<float> in = {3,5,7,11,13,17,19,23,29,31};
            tensor<float> gpuIn(ctx, 1, 5, 1, 2, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({3});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {8.0f,18.0f,30.0f,42.0f,60.0f}, "Reduce Sum 2d Long");
        }

        {
            std::vector<float> in = {3,5,7,11,13,17,19,23,29,31};
            tensor<float> gpuIn(ctx, 5, 1, 1, 2, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({3});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {8.0f,18.0f,30.0f,42.0f,60.0f}, "Reduce Sum 2d Long");
        }


        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};
            tensor<float> gpuIn(ctx, 1, 2, 5, 1, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({2});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {10.0f, 35.0f}, "Reduce Sum 2d Width");
        }

        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};
            tensor<float> gpuIn(ctx, 1, 5, 2, 1, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({1});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {20.0f, 25.0f}, "Reduce Sum 2d Height");
        }

        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};
            tensor<float> gpuIn(ctx, 1, 2, 5, 1, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({1});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {5.0f,7.0f,9.0f,11.0f,13.0f}, "Reduce Sum 2d Long");
        }

        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};
            tensor<float> gpuIn(ctx, 5, 2, 1, 1, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceSum({0});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {20.0f, 25.0f}, "Reduce Sum 2d Number");
        }
    }

    void testReduceMax2D()
    {
        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};
            tensor<float> gpuIn(ctx, 1, 1, 2, 5, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceMax({3});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {4.0f, 9.0f}, "Reduce Max 2d Depth");
        }

        {
            std::vector<float> in = {0,1,2,3,4,5,6,7,8,9};
            tensor<float> gpuIn(ctx, 1, 1, 5, 2, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceMax({2});
            std::vector<float> cpuOut = temp.toCpu();

            expect(cpuOut, {8.0f, 9.0f}, "Reduce Max 2d Depth");
        }

        // {
        //     std::vector<float> in(1024*2048*3);
        //     float max = -10000;
        //     for(int i = 0; i < (int)in.size(); i++)
        //     {
        //         in[i] = (int)(i/1000);
        //         if(in[i] > max)
        //             max = in[i];
        //     }
        //     tensor<float> gpuIn(ctx, 1, 1024, 2048, 3, true);
        //     gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

        //     auto temp = gpuIn.reduceMaxAll();
        //     std::vector<float> cpuOut = temp.toCpu();

        //     expect(cpuOut, {max}, "Reduce Max Large");
        // }

        // {
        //     std::vector<float> in(1024*2048*50);
        //     float max = -10000;
        //     for(int i = 0; i < (int)in.size(); i++)
        //     {
        //         if(i % (1024*256*10) == 0)
        //         {
        //             in[i] = (int)(i/10000);
        //             if(in[i] > max)
        //                 max = in[i];
        //         }
        //     }
        //     tensor<float> gpuIn(ctx, 1, 1024, 2048, 50, true);
        //     gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

        //     auto temp = gpuIn.reduceMaxAll();
        //     std::vector<float> cpuOut = temp.toCpu();

        //     expect(cpuOut, {max}, "Reduce Max Large Sparse");
        // }

        // {
        //     std::vector<double> in(1024*2048*50);
        //     double max = -10000;
        //     for(int i = 0; i < (int)in.size(); i++)
        //     {
        //         if(i % (1024*256*10) == 0)
        //         {
        //             in[i] = (int)(i/10000);
        //             if(in[i] > max)
        //                 max = in[i];
        //         }
        //     }
        //     tensor<double> gpuIn(ctx, 1, 1024, 2048, 50, true);
        //     gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

        //     auto temp = gpuIn.reduceMaxAll();
        //     std::vector<double> cpuOut = temp.toCpu();

        //     expect(cpuOut, {max}, "Reduce Max Large Sparse Double");

        // }

        {
            std::vector<double> in(10*10*3);
            double max = -10000;
            for(int i = 0; i < (int)in.size(); i++)
            {
                in[i] = (int)(i/2);
                if(in[i] > max)
                    max = in[i];
            }
            tensor<double> gpuIn(ctx, 1, 10, 10, 3, true);
            gpuIn.toGpu(in, CUDNN_TENSOR_NHWC);

            auto temp = gpuIn.reduceMaxAll();
            std::vector<double> cpuOut = temp.toCpu();

            expect(cpuOut, {max}, "Reduce Max Double");
        }
    }

    void runAllTests()
    {
        testAdd();
        testAlloc();
        testReduceSum1D();
        testReduceSum2D();
        testReduceMax2D();
    }
};
