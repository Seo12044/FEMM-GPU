cmake_minimum_required(VERSION 3.18)

# Kept separate from the legacy MFC targets so a normal FEMM build never
# requires a CUDA toolkit. Leave empty to use the CUDA toolchain default.
set(GPU_SOLVER_CUDA_ARCHITECTURES "" CACHE STRING
    "CUDA architectures for gpu_linear_p1_poc (empty uses toolchain default)")

enable_language(CUDA)
find_package(Threads REQUIRED)

add_executable(gpu_linear_p1_poc
  src/linear_p1_poc.cu
  src/native_airgap_matrix.cpp
)
# Keep the historical executable name for the MATLAB integration and publish
# a neutral name for standalone users.  Both files are byte-identical.
add_custom_command(TARGET gpu_linear_p1_poc POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:gpu_linear_p1_poc>
    $<TARGET_FILE_DIR:gpu_linear_p1_poc>/femm_gpu$<TARGET_FILE_SUFFIX:gpu_linear_p1_poc>
  VERBATIM
)
# Keep the GPU AGE stiffness matrix bit-for-bit derived from FEMM's native
# fkn implementation.  It is generated at configure time solely from the
# tracked source, so a future FEMM correction cannot silently diverge here.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../fkn/prob1big.cpp" GPU_FEMM_FKN_SOURCE)
string(REGEX MATCH "MG\\[0\\]\\[0\\] =.*// Add each annulus element to the global stiffness matrix"
  GPU_FEMM_AGE_MATRIX_SOURCE "${GPU_FEMM_FKN_SOURCE}")
if(NOT GPU_FEMM_AGE_MATRIX_SOURCE)
  message(FATAL_ERROR "Could not extract FEMM native air-gap element matrix")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/gpu_femm_native_age_matrix.inc"
  "${GPU_FEMM_AGE_MATRIX_SOURCE}")
target_include_directories(gpu_linear_p1_poc PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(gpu_linear_p1_poc PRIVATE Threads::Threads)
# This target uses CUDA cooperative groups.  The existing FEMM-derived source
# relies on the legacy MSVC preprocessor's token handling, so acknowledge the
# CUDA header warning without changing the compiler mode for legacy code.
target_compile_definitions(gpu_linear_p1_poc PRIVATE
  CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
)

add_executable(gpu_bh_curve_poc
  src/bh_curve_poc.cu
)

# FEMM's preprocessor generates the exact Triangle mesh before launching
# fkn.exe.  The MATLAB GPU adapter stages a private copy of stock FEMM and
# installs this no-op executable as that copy's fkn.exe, preserving the mesh
# files without running the CPU magnetic solver or changing stock FEMM.
if(WIN32)
  add_executable(gpu_femm_mesh_noop_solver WIN32
    src/mesh_noop_solver.cpp
  )
else()
  add_executable(gpu_femm_mesh_noop_solver
    src/mesh_noop_solver.cpp
  )
endif()

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

set_target_properties(gpu_femm_mesh_noop_solver PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
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
  find_package(Python3 COMPONENTS Interpreter QUIET)
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
  add_test(NAME gpu_nonlinear_p1_postprocess_reference
    COMMAND $<TARGET_FILE:gpu_linear_p1_poc> --postprocess-reference
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/nonlinear_pm_coil_v1/nonlinear_pm_coil_planar"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/35PN230_v1"
  )
  add_test(NAME gpu_nonlinear_p1_single_sample_adapter
    COMMAND $<TARGET_FILE:gpu_linear_p1_poc> --single-sample
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/nonlinear_pm_coil_v1/single_sample_request.json"
      "${CMAKE_CURRENT_BINARY_DIR}/nonlinear_pm_coil_single_sample_response.json"
  )
  set_tests_properties(gpu_nonlinear_p1_single_sample_adapter PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  )
  add_test(NAME gpu_bh_curve_35pn230
    COMMAND $<TARGET_FILE:gpu_bh_curve_poc> --fixture
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/35PN230_v1"
  )
  add_test(NAME gpu_femm_mesh_noop_solver
    COMMAND $<TARGET_FILE:gpu_femm_mesh_noop_solver> ignored-mesh-stem
  )
  if(Python3_Interpreter_FOUND)
    add_test(NAME gpu_femm_project_neutral_cli
      COMMAND ${Python3_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_femm_gpu_cli.py"
        $<TARGET_FILE:gpu_linear_p1_poc>
    )
    add_test(NAME gpu_femm_project_neutral_prepare
      COMMAND ${Python3_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_femm_gpu_prepare.py"
    )
  endif()
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
  set_tests_properties(gpu_nonlinear_p1_postprocess_reference
    gpu_nonlinear_p1_single_sample_adapter PROPERTIES
    LABELS "gpu;requires_cuda;femm_fixture;nonlinear;postprocess"
    TIMEOUT 30
  )
  set_tests_properties(gpu_bh_curve_35pn230 PROPERTIES
    LABELS "gpu;requires_cuda;bh_curve"
    TIMEOUT 30
  )
  set_tests_properties(gpu_femm_mesh_noop_solver PROPERTIES
    LABELS "gpu;mesh_only"
    TIMEOUT 5
  )
  if(Python3_Interpreter_FOUND)
    set_tests_properties(gpu_femm_project_neutral_cli
      gpu_femm_project_neutral_prepare PROPERTIES
      LABELS "gpu;cli;no_cuda"
      TIMEOUT 10
    )
  endif()
endif()
