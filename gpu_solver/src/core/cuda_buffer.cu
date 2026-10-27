#include "cuda_buffer.cuh"

namespace gpu_femm
{

Status CopyToDevice(void *destination, const void *source, size_t bytes)
{
  if (bytes == 0)
    return Status::kOk;
  return cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice) == cudaSuccess
             ? Status::kOk
             : Status::kInternalError;
}

Status CopyToHost(void *destination, const void *source, size_t bytes)
{
  if (bytes == 0)
    return Status::kOk;
  return cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost) == cudaSuccess
             ? Status::kOk
             : Status::kInternalError;
}

} // namespace gpu_femm
