cmake_minimum_required(VERSION 2.8)
project(poisson)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6.0
    OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 6.0)
  message(FATAL_ERROR "CUDA 8.0 does not support gcc6+. Please run:\n
cmake -DCMAKE_C_COMPILER=`which gcc-5` -DCMAKE_CXX_COMPILER=`which g++-5` ..")
endif()
endif()

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 14)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  list(APPEND CMAKE_CXX_FLAGS "-std=gnu++14")
endif ()

find_package(CUDA)
include(FindCUDA)
include_directories(${CUDA_INCLUDE_DIRS})

list(APPEND CUDA_NVCC_FLAGS "-std=c++11 -gencode arch=compute_30,code=compute_30") #--ftz=false") # || --use_fast_math
set(CUDA_PROPAGATE_HOST_FLAGS OFF)


if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND CUDA_NVCC_FLAGS "-g -lineinfo")
endif()

cuda_add_executable(${PROJECT_NAME} poisson.cu)
target_link_libraries(${PROJECT_NAME}
  ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY}
)
