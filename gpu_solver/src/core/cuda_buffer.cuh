#pragma once

#include "types.h"
#include <cuda_runtime.h>

// CUDA/MSVC compatibility headers reserve ordinary FEMM local names.
#undef element
#undef near
#undef far

namespace gpu_femm
{

template <typename T> class DeviceBuffer
{
public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { reset(); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  Status allocate(size_t count)
  {
    reset();
    count_ = count;
    if (count == 0)
      return Status::kOk;
    if (cudaMalloc(&ptr_, count * sizeof(T)) != cudaSuccess) {
      ptr_ = nullptr;
      count_ = 0;
      return Status::kGpuAllocationFailed;
    }
    return Status::kOk;
  }

  void reset()
  {
    if (ptr_ != nullptr)
      cudaFree(ptr_);
    ptr_ = nullptr;
    count_ = 0;
  }

  T *get() { return ptr_; }
  const T *get() const { return ptr_; }
  size_t count() const { return count_; }

private:
  T *ptr_ = nullptr;
  size_t count_ = 0;
};

Status CopyToDevice(void *destination, const void *source, size_t bytes);
Status CopyToHost(void *destination, const void *source, size_t bytes);

template <typename T> class PinnedBuffer
{
public:
  PinnedBuffer() = default;
  ~PinnedBuffer() { reset(); }
  PinnedBuffer(const PinnedBuffer &) = delete;
  PinnedBuffer &operator=(const PinnedBuffer &) = delete;

  Status allocate(size_t count)
  {
    reset();
    if (count && cudaMallocHost(&ptr_, count * sizeof(T)) != cudaSuccess) {
      ptr_ = nullptr;
      return Status::kGpuAllocationFailed;
    }
    return Status::kOk;
  }
  T *get() { return ptr_; }
  void reset()
  {
    if (ptr_)
      cudaFreeHost(ptr_);
    ptr_ = nullptr;
  }

private:
  T *ptr_ = nullptr;
};

} // namespace gpu_femm
