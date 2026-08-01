cmake_minimum_required(VERSION 3.18)

# Kept separate from the legacy MFC targets so a normal FEMM build never
# requires a CUDA toolkit. Leave empty to use the CUDA toolchain default.
set(GPU_SOLVER_CUDA_ARCHITECTURES "" CACHE STRING
    "CUDA architectures for gpu_linear_p1_poc (empty uses toolchain default)")

enable_language(CUDA)

add_executable(gpu_linear_p1_poc
  src/linear_p1_poc.cu
)

add_executable(gpu_bh_curve_poc
  src/bh_curve_poc.cu
)

set_target_properties(gpu_linear_p1_poc PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
)

set_target_properties(gpu_bh_curve_poc PROPERTIES
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
  target_compile_options(gpu_bh_curve_poc PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
  )
endif()

# CMAKE_CUDA_ARCHITECTURES remains usable as the standard CMake override.
if(GPU_SOLVER_CUDA_ARCHITECTURES)
  set_target_properties(gpu_linear_p1_poc PROPERTIES
    CUDA_ARCHITECTURES "${GPU_SOLVER_CUDA_ARCHITECTURES}"
  )
  set_target_properties(gpu_bh_curve_poc PROPERTIES
    CUDA_ARCHITECTURES "${GPU_SOLVER_CUDA_ARCHITECTURES}"
  )
endif()

if(BUILD_TESTING)
  add_test(NAME gpu_linear_p1_selftest
    COMMAND $<TARGET_FILE:gpu_linear_p1_poc> --self-test
  )
  add_test(NAME gpu_linear_p1_femm_reference
    COMMAND $<TARGET_FILE:gpu_linear_p1_poc> --femm-reference
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/linear_square_v1/linear_square"
  )
  add_test(NAME gpu_nonlinear_p1_femm_reference
    COMMAND $<TARGET_FILE:gpu_linear_p1_poc> --nonlinear-reference
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/nonlinear_pm_coil_v1/nonlinear_pm_coil_planar"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/35PN230_v1"
  )
  add_test(NAME gpu_bh_curve_35pn230
    COMMAND $<TARGET_FILE:gpu_bh_curve_poc> --fixture
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/35PN230_v1"
  )
  set_tests_properties(gpu_linear_p1_selftest PROPERTIES
    LABELS "gpu;requires_cuda"
    TIMEOUT 30
  )
  set_tests_properties(gpu_linear_p1_femm_reference PROPERTIES
    LABELS "gpu;requires_cuda;femm_fixture"
    TIMEOUT 30
  )
  set_tests_properties(gpu_nonlinear_p1_femm_reference PROPERTIES
    LABELS "gpu;requires_cuda;femm_fixture;nonlinear"
    TIMEOUT 30
  )
  set_tests_properties(gpu_bh_curve_35pn230 PROPERTIES
    LABELS "gpu;requires_cuda;bh_curve"
    TIMEOUT 30
  )
endif()
