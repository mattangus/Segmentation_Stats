cmake_minimum_required(VERSION 3.0)

project(SegStats)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "--std=c++11 -Wall -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -D GOOGLE_CUDA=1")
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
target_link_libraries(SegStats ${OpenCV_LIBS} ${CUDA_LIBRARIES} stdc++fs)