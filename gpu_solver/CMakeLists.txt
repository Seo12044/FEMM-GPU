cmake_minimum_required(VERSION 3.18)

# Kept separate from the legacy MFC targets so a normal FEMM build never
# requires a CUDA toolkit. Leave empty to use the CUDA toolchain default.
set(GPU_SOLVER_CUDA_ARCHITECTURES "" CACHE STRING
    "CUDA architectures for gpu_linear_p1_poc (empty uses toolchain default)")

enable_language(CUDA)

add_executable(gpu_linear_p1_poc
  src/linear_p1_poc.cu
)

set_target_properties(gpu_linear_p1_poc PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
)

# Forward UTF-8 handling only to the MSVC host compiler used by NVCC. This
# prevents C4819 noise from NVIDIA headers without changing legacy targets.
if(MSVC)
  target_compile_options(gpu_linear_p1_poc PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
  )
endif()

# CMAKE_CUDA_ARCHITECTURES remains usable as the standard CMake override.
if(GPU_SOLVER_CUDA_ARCHITECTURES)
  set_target_properties(gpu_linear_p1_poc PROPERTIES
    CUDA_ARCHITECTURES "${GPU_SOLVER_CUDA_ARCHITECTURES}"
  )
endif()

if(BUILD_TESTING)
  add_test(NAME gpu_linear_p1_selftest
    COMMAND $<TARGET_FILE:gpu_linear_p1_poc> --self-test
  )
  set_tests_properties(gpu_linear_p1_selftest PROPERTIES
    LABELS "gpu;requires_cuda"
    TIMEOUT 30
  )
endif()
