cmake_minimum_required(VERSION 3.18)
project(speculative_decoding LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)

message(STATUS "Building for CUDA architecture: sm_${CMAKE_CUDA_ARCHITECTURES}")

add_executable(speculative_decoding
    src/main.cu
    src/verification.cu
    src/residual_sample.cu
    src/benchmark.cu
    src/cpu_baseline.cpp
    src/npy_loader.cpp
    src/pinned_benchmark.cu
    src/stream_pipeline.cu
)

target_include_directories(speculative_decoding PRIVATE src)
target_link_libraries(speculative_decoding PRIVATE curand)

set_target_properties(speculative_decoding PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

if(MSVC)
    target_compile_options(speculative_decoding PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3>
    )
else()
    target_compile_options(speculative_decoding PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    )
endif()