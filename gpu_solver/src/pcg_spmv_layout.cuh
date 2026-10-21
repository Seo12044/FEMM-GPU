// Internal storage for row-ordered Sliced ELL. No row sorting or arithmetic
// reordering: each row consumes exactly its original CSR entries, in order.
constexpr int kPcgSliceRows = 32;

__global__ void PackPcgValues(int nnz, int slots, const int32_t* source,
    const double* csr, double* packed)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= slots) return;
  const size_t item = blockIdx.y;
  packed[item * slots + index] = source[index] < 0
      ? 0.0 : csr[item * nnz + source[index]];
}

class PcgSpmvLayout {
 public:
  void Reset()
  {
    nnz_ = slots_ = 0;
    slices_.reset(); source_.reset(); columns_.reset(); values_.reset();
  }

  Status Initialize(const std::vector<int32_t>& rows,
      const std::vector<int32_t>& columns, bool enabled)
  {
    Reset();
    const size_t n = rows.size() - 1;
    // Tiny matrices do not amortize packing. Irregular slices retain CSR
    // storage locally, so a few long AGE rows cannot disable all other slices.
    if (!enabled || n < 512 || columns.empty()) return Status::kOk;
    std::vector<int32_t> slices;
    size_t slots = 0;
    for (size_t first = 0; first < n; first += kPcgSliceRows) {
      int width = 0;
      const size_t end = std::min(n, first + kPcgSliceRows);
      for (size_t row = first; row < end; ++row)
        width = std::max(width, rows[row + 1] - rows[row]);
      const size_t csr_entries = rows[end] - rows[first];
      const size_t padded = static_cast<size_t>(width) * kPcgSliceRows;
      const bool packed = padded <= 2 * csr_entries;
      slices.push_back(packed ? static_cast<int32_t>(slots) : -1 - static_cast<int32_t>(slots));
      slots += packed ? padded : csr_entries;
      if (slots > static_cast<size_t>(INT_MAX - kPcgSliceRows))
        return Status::kOk;
    }
    std::vector<int32_t> source(slots, -1), packed_columns(slots, 0);
    for (size_t row = 0; row < n; ++row) {
      const int32_t slice = slices[row / kPcgSliceRows];
      size_t index = slice < 0 ? (-1 - slice) + (rows[row] - rows[row / kPcgSliceRows * kPcgSliceRows])
          : slice + row % kPcgSliceRows;
      const int stride = slice < 0 ? 1 : kPcgSliceRows;
      for (int entry = rows[row]; entry < rows[row + 1]; ++entry, index += stride) {
        source[index] = entry;
        packed_columns[index] = columns[entry];
      }
    }
    Status status = Status::kOk;
    if ((status = slices_.allocate(slices.size())) != Status::kOk
        || (status = source_.allocate(slots)) != Status::kOk
        || (status = columns_.allocate(slots)) != Status::kOk
        || (status = CopyToDevice(slices_.get(), slices.data(), slices.size() * sizeof(int32_t))) != Status::kOk
        || (status = CopyToDevice(source_.get(), source.data(), slots * sizeof(int32_t))) != Status::kOk
        || (status = CopyToDevice(columns_.get(), packed_columns.data(), slots * sizeof(int32_t))) != Status::kOk) {
      Reset();
      return status;
    }
    nnz_ = static_cast<int>(columns.size());
    slots_ = static_cast<int>(slots);
    return Status::kOk;
  }

  Status Pack(size_t count, const double* csr)
  {
    if (!enabled()) return Status::kOk;
    if (count == 0 || count > 65535) return Status::kInvalidArgument;
    const size_t entries = count * static_cast<size_t>(slots_);
    // Bound the retained buffer to this batch, not the largest campaign batch.
    if (values_.count() != entries) {
      const Status status = values_.allocate(entries);
      if (status != Status::kOk) return status;
    }
    PackPcgValues<<<dim3((slots_ + 255u) / 256u, static_cast<unsigned>(count)), 256>>>(
        nnz_, slots_, source_.get(), csr, values_.get());
    return cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess
        ? Status::kOk : Status::kInternalError;
  }

  bool enabled() const { return slots_ != 0; }
  int slots() const { return slots_; }
  const int32_t* slices() const { return slices_.get(); }
  const int32_t* columns() const { return columns_.get(); }
  const double* values() const { return values_.get(); }
  size_t numeric_bytes() const { return values_.count() * sizeof(double); }

 private:
  int nnz_ = 0, slots_ = 0;
  DeviceBuffer<int32_t> slices_, source_, columns_;
  DeviceBuffer<double> values_;
};

__device__ double PcgRowDot(int row, const int32_t* rows,
    const int32_t* columns, const double* values, const double* vector,
    const int32_t* slices)
{
  double sum = 0.0;
  if (slices) {
    const int slice = slices[row / kPcgSliceRows];
    int index = slice < 0 ? (-1 - slice) + (rows[row] - rows[row / kPcgSliceRows * kPcgSliceRows])
        : slice + row % kPcgSliceRows;
    const int stride = slice < 0 ? 1 : kPcgSliceRows;
    for (int entry = rows[row]; entry < rows[row + 1]; ++entry, index += stride)
      sum += values[index] * vector[columns[index]];
  } else {
    for (int entry = rows[row]; entry < rows[row + 1]; ++entry)
      sum += values[entry] * vector[columns[entry]];
  }
  return sum;
}
