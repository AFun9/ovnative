# CUDA backend (phase P6).
#
# Only built when -DOV_ENABLE_CUDA=ON and the CUDA toolkit is present.  The
# public headers of the engine stay CUDA-free; this directory owns every
# `__global__` symbol and the cuBLAS handle.
add_library(ovcuda STATIC
  cuda_lm.cu
  cuda_decoder.cu
  cuda_encoder.cu
)
target_include_directories(ovcuda PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
target_link_libraries(ovcuda PUBLIC ovcore)
set_target_properties(ovcuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                       CUDA_STANDARD 17
                                       POSITION_INDEPENDENT_CODE ON)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(ovcuda PUBLIC CUDA::cudart CUDA::cublas)
target_compile_options(ovcuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
