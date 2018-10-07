#include <vector>
#include <cudnn.h>
#include <cuda.h>

#include "testHelper.cuh"
#include "tensorTests.cuh"

int main(int argc, char **argv) {
    std::cout << "cudnn ver: " << CUDNN_MAJOR << "." << CUDNN_MINOR << "." << CUDNN_PATCHLEVEL << std::endl;

    globalSetup();

    tensorTests::runAllTests();

    return 0;
}