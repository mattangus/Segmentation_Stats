cmake_minimum_required(VERSION 3.0)

project(SegStats)

enable_testing()

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
#find_package(CUDNN REQUIRED)

set(CMAKE_CXX_FLAGS "--std=c++11 -Wall -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -D GOOGLE_CUDA=1")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --default-stream per-thread" )
set(CMAKE_BUILD_TYPE Debug)

#pass flags to c++ compiler
set(CUDA_PROPAGATE_HOST_FLAGS ON)

set(MAIN_SRC
	"src/main.cu"
)
include_directories("include")
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

cuda_add_executable(SegStats ${MAIN_SRC})
target_link_libraries(SegStats ${OpenCV_LIBS} ${CUDA_LIBRARIES} cudnn stdc++fs)


set(TEST_SRC
	"test/runTests.cu"
)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

cuda_add_executable(runTests ${TEST_SRC})
target_link_libraries(runTests ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${GTEST_BOTH_LIBRARIES} cudnn stdc++fs)