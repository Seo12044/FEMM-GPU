#pragma once

#include "../core/types.h"

namespace gpu_femm
{

// The two host specializations are instantiated in pcg_kernels.cu.
template <bool kUseInverseDiagonal>
void LaunchDeterministicPcg(unsigned count, const double *blocks, const int *owners,
                            const int32_t *slices, int n, const int32_t *row_offsets,
                            const int32_t *column_indices, const double *values,
                            const double *diagonal, const double *inverse_diagonal,
                            const double *rhs, double *x, double *residual, double *direction,
                            double *preconditioned, double *matrix_direction,
                            double relative_tolerance, int max_iterations, SolveInfo *info,
                            int values_per_item);

// Resolve the kernel in its defining CUDA translation unit so callers use a
// host entry point rather than referencing CUDA template-kernel stubs.
const void *CooperativePcgKernel(bool use_inverse_diagonal, bool block_preconditioner);

} // namespace gpu_femm
