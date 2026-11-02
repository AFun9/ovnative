cmake_minimum_required(VERSION 3.16)
project(ovnative LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(OV_ENABLE_OPENMP "Parallelise the GEMM over row blocks" ON)
option(OV_ENABLE_CUDA   "Build the CUDA backend (see src/backend/cuda/README.md)" OFF)

if(MSVC)
  # MSVC does not define M_PI from <cmath> unless asked.
  add_compile_definitions(_USE_MATH_DEFINES NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()

# ---------------------------------------------------------------- core library
add_library(ovcore STATIC
  src/core/weight_file.cpp
  src/backend/cpu/gemm.cpp
  src/backend/cpu/ops_cpu.cpp
  src/backend/cpu/conv_cpu.cpp
  src/model/mat.cpp
  src/model/lm.cpp
  src/model/audio_tokenizer.cpp
  src/model/tts.cpp
  src/model/at_encoder.cpp
  src/model/audio_io.cpp
  src/tokenizer/bpe.cpp
)
target_include_directories(ovcore PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
if(MSVC)
  target_compile_options(ovcore PRIVATE /W3 /utf-8)
  # The AVX2 kernels are compiled with /arch:AVX2, so no runtime dispatch is
  # needed on Windows. AVX-VNNI is disabled on MSVC (see gemm.cpp), and the CPU
  # path falls back to the AVX2/legacy int8 GEMM.
  set_source_files_properties(
    src/backend/cpu/gemm.cpp
    src/backend/cpu/ops_cpu.cpp
    PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  target_compile_options(ovcore PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

if(OV_ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX QUIET)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(ovcore PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP: enabled")
  else()
    message(STATUS "OpenMP: not found, building single-threaded")
  endif()
endif()

# ------------------------------------------------------------------ C ABI lib
add_library(ovnative_c STATIC src/api/ovnative_c.cpp)
target_link_libraries(ovnative_c PUBLIC ovcore)
set_target_properties(ovnative_c PROPERTIES OUTPUT_NAME ovnative)

# ----------------------------------------------------------------------- tools
add_executable(ovdump tools/ovdump.cpp)
target_link_libraries(ovdump PRIVATE ovcore)

add_executable(ovdecode tools/ovdecode.cpp)
target_link_libraries(ovdecode PRIVATE ovcore)

add_executable(ovtts tools/ovtts.cpp)
target_link_libraries(ovtts PRIVATE ovcore)
if(WIN32)
  target_link_libraries(ovtts PRIVATE shell32)  # CommandLineToArgvW
endif()

add_executable(ovweb tools/ovweb.cpp)
target_link_libraries(ovweb PRIVATE ovcore)
if(WIN32)
  target_link_libraries(ovweb PRIVATE shell32 ws2_32)
endif()

# ----------------------------------------------------------------------- tests
enable_testing()
add_executable(ov_test_gemm tests/test_gemm.cpp)
target_link_libraries(ov_test_gemm PRIVATE ovcore)
add_test(NAME gemm COMMAND ov_test_gemm)

add_executable(ov_test_lm tests/test_lm.cpp)
target_link_libraries(ov_test_lm PRIVATE ovcore)

add_executable(ov_test_at tests/test_at_decoder.cpp)
target_link_libraries(ov_test_at PRIVATE ovcore)

add_executable(ov_test_tokenizer tests/test_tokenizer.cpp)
target_link_libraries(ov_test_tokenizer PRIVATE ovcore)

add_executable(ov_test_at_enc tests/test_at_encoder.cpp)
target_link_libraries(ov_test_at_enc PRIVATE ovcore)

add_executable(ov_test_c_api tests/test_c_api.cpp)
target_link_libraries(ov_test_c_api PRIVATE ovnative_c)

# The LM gate needs generated artefacts; register it only when they are present.
set(OV_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(OV_LM_FP32 ${OV_REPO_ROOT}/output/weights/omnivoice-lm-fp32.ovw)
set(OV_LM_INT8 ${OV_REPO_ROOT}/output/weights/omnivoice-lm-int8.ovw)
if(EXISTS ${OV_LM_FP32})
  add_test(NAME lm_fp32 COMMAND ov_test_lm ${OV_LM_FP32} ${OV_REPO_ROOT}/output/ref/lm/b1_s64_full)
endif()
if(EXISTS ${OV_LM_INT8})
  add_test(NAME lm_int8 COMMAND ov_test_lm ${OV_LM_INT8} ${OV_REPO_ROOT}/output/ref/lm/b1_s64_full)
endif()
# batch > 1 exercises the batched paths (score planes, arena sizing) that a
# single-sequence reference cannot reach.
set(OV_LM_B2 ${OV_REPO_ROOT}/output/ref/lm/b2_s256_block)
if(EXISTS ${OV_LM_B2} AND EXISTS ${OV_LM_FP32})
  add_test(NAME lm_fp32_b2 COMMAND ov_test_lm ${OV_LM_FP32} ${OV_LM_B2})
endif()
set(OV_TOK ${OV_REPO_ROOT}/output/weights/tokenizer.ovt)
if(EXISTS ${OV_TOK})
  add_test(NAME tokenizer COMMAND ov_test_tokenizer ${OV_TOK} ${OV_REPO_ROOT}/output/ref/tokenizer/cases.bin)
  add_test(NAME c_api COMMAND ov_test_c_api ${OV_LM_INT8} ${OV_REPO_ROOT}/output/weights/omnivoice-at-fp32.ovw ${OV_TOK})
endif()
set(OV_AT_FP32 ${OV_REPO_ROOT}/output/weights/omnivoice-at-fp32.ovw)
if(EXISTS ${OV_AT_FP32})
  add_test(NAME at_decoder_fp32 COMMAND ov_test_at ${OV_AT_FP32} ${OV_REPO_ROOT}/output/ref/at_dec/f50)
  if(EXISTS ${OV_REPO_ROOT}/output/ref/at_enc/semantic_downsampled.f32)
    add_test(NAME at_encoder_hubert COMMAND ov_test_at_enc ${OV_AT_FP32} ${OV_REPO_ROOT}/output/ref/at_enc ${OV_REPO_ROOT}/assets/zero_shot_prompt.wav)
  endif()
endif()

if(OV_ENABLE_CUDA)
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)  # Ada (RTX 4090); override for other GPUs
  endif()
  add_subdirectory(src/backend/cuda)
  # Let the CLI / tests use the accelerator without making ovcore depend on CUDA.
  target_compile_definitions(ovtts PRIVATE OV_ENABLE_CUDA=1)
  target_link_libraries(ovtts PRIVATE ovcuda)
  target_compile_definitions(ovweb PRIVATE OV_ENABLE_CUDA=1)
  target_link_libraries(ovweb PRIVATE ovcuda)
  target_compile_definitions(ov_test_lm PRIVATE OV_ENABLE_CUDA=1)
  target_link_libraries(ov_test_lm PRIVATE ovcuda)
  if(EXISTS ${OV_LM_FP32})
    add_test(NAME lm_fp32_cuda
             COMMAND ov_test_lm ${OV_LM_FP32} ${OV_REPO_ROOT}/output/ref/lm/b1_s64_full --cuda)
  endif()
  if(EXISTS ${OV_LM_B2})
    add_test(NAME lm_fp32_cuda_b2 COMMAND ov_test_lm ${OV_LM_FP32} ${OV_LM_B2} --cuda)
  endif()
endif()
