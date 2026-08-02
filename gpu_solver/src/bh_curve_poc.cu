#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace gpu_femm {

struct BhEvaluation {
  double h_a_per_m;
  double dh_db;
  double reluctivity_m_per_h;
  double dv_db2;
};

struct BhCurve {
  std::vector<double> b_t;
  std::vector<double> h_a_per_m;
  std::vector<double> slope;
  int smoothing_passes = 0;
};

bool Finite(double value) { return std::isfinite(value); }

bool ValidateInput(const BhCurve& curve, std::string* error)
{
  if (curve.b_t.size() < 2 || curve.b_t.size() != curve.h_a_per_m.size()) {
    *error = "B-H curve requires at least two paired points";
    return false;
  }
  if (curve.b_t.front() != 0.0 || curve.h_a_per_m.front() != 0.0) {
    *error = "B-H curve must start at the origin";
    return false;
  }
  for (size_t i = 0; i < curve.b_t.size(); ++i) {
    if (!Finite(curve.b_t[i]) || !Finite(curve.h_a_per_m[i])
        || curve.b_t[i] < 0.0 || curve.h_a_per_m[i] < 0.0) {
      *error = "B-H curve contains a nonfinite or negative value";
      return false;
    }
    if (i > 0 && !(curve.b_t[i] > curve.b_t[i - 1])) {
      *error = "B values must be strictly increasing";
      return false;
    }
  }
  return true;
}

bool DenseGaussSolve(std::vector<double>* matrix, std::vector<double>* rhs)
{
  const int n = static_cast<int>(rhs->size());
  for (int i = 0; i < n; ++i) {
    int pivot = i;
    double maximum = 0.0;
    for (int row = i; row < n; ++row) {
      const double candidate = std::abs((*matrix)[row * n + i]);
      if (candidate > maximum) {
        maximum = candidate;
        pivot = row;
      }
    }
    if (maximum == 0.0 || !Finite(maximum))
      return false;
    if (pivot != i) {
      for (int column = 0; column < n; ++column)
        std::swap((*matrix)[i * n + column], (*matrix)[pivot * n + column]);
      std::swap((*rhs)[i], (*rhs)[pivot]);
    }
    for (int row = i + 1; row < n; ++row) {
      const double factor = (*matrix)[row * n + i] / (*matrix)[i * n + i];
      (*rhs)[row] -= factor * (*rhs)[i];
      for (int column = i; column < n; ++column)
        (*matrix)[row * n + column] -= factor * (*matrix)[i * n + column];
    }
  }
  for (int row = n - 1; row >= 0; --row) {
    double sum = 0.0;
    for (int column = n - 1; column > row; --column)
      sum += (*matrix)[row * n + column] * (*rhs)[column];
    (*rhs)[row] = ((*rhs)[row] - sum) / (*matrix)[row * n + row];
    if (!Finite((*rhs)[row]))
      return false;
  }
  return true;
}

bool ComputeFemmDcSlopes(BhCurve* curve, std::string* error)
{
  if (!ValidateInput(*curve, error))
    return false;
  const int n = static_cast<int>(curve->b_t.size());
  constexpr int kMaxSmoothingPasses = 1024;
  for (int pass = 0; pass <= kMaxSmoothingPasses; ++pass) {
    std::vector<double> matrix(static_cast<size_t>(n) * n, 0.0);
    std::vector<double> rhs(n, 0.0);
    double length = curve->b_t[1] - curve->b_t[0];
    matrix[0] = 4.0 / length;
    matrix[1] = 2.0 / length;
    rhs[0] = 6.0 * (curve->h_a_per_m[1] - curve->h_a_per_m[0])
        / (length * length);
    length = curve->b_t[n - 1] - curve->b_t[n - 2];
    matrix[(n - 1) * n + (n - 1)] = 4.0 / length;
    matrix[(n - 1) * n + (n - 2)] = 2.0 / length;
    rhs[n - 1] = 6.0 * (curve->h_a_per_m[n - 1] - curve->h_a_per_m[n - 2])
        / (length * length);
    for (int i = 1; i < n - 1; ++i) {
      const double left = curve->b_t[i] - curve->b_t[i - 1];
      const double right = curve->b_t[i + 1] - curve->b_t[i];
      matrix[i * n + i - 1] = 2.0 / left;
      matrix[i * n + i] = 4.0 * (left + right) / (left * right);
      matrix[i * n + i + 1] = 2.0 / right;
      rhs[i] = 6.0 * (curve->h_a_per_m[i] - curve->h_a_per_m[i - 1])
              / (left * left)
          + 6.0 * (curve->h_a_per_m[i + 1] - curve->h_a_per_m[i])
              / (right * right);
    }
    if (!DenseGaussSolve(&matrix, &rhs)) {
      *error = "FEMM B-H slope system is singular";
      return false;
    }
    curve->slope = rhs;

    bool curve_ok = true;
    for (int i = 1; i < n; ++i) {
      const double segment = curve->b_t[i] - curve->b_t[i - 1];
      const double d0 = curve->slope[i - 1];
      const double d1 = curve->slope[i];
      const double h0 = curve->h_a_per_m[i - 1];
      const double h1 = curve->h_a_per_m[i];
      const double c0 = d0;
      const double c1 = -2.0 * (2.0 * d0 * segment + d1 * segment
          + 3.0 * h0 - 3.0 * h1) / (segment * segment);
      const double c2 = 3.0 * (d0 * segment + d1 * segment
          + 2.0 * h0 - 2.0 * h1) / (segment * segment * segment);
      double x0 = -1.0;
      double x1 = -1.0;
      const double discriminant = c1 * c1 - 4.0 * c0 * c2;
      if (c2 == 0.0) {
        if (c1 != 0.0)
          x0 = -c0 / c1;
      } else if (discriminant > 0.0) {
        const double root = std::sqrt(discriminant);
        x0 = -(c1 + root) / (2.0 * c2);
        x1 = (-c1 + root) / (2.0 * c2);
      }
      if ((x0 >= 0.0 && x0 <= segment) || (x1 >= 0.0 && x1 <= segment))
        curve_ok = false;
    }
    if (curve_ok) {
      curve->smoothing_passes = pass;
      return true;
    }
    if (pass == kMaxSmoothingPasses) {
      *error = "FEMM B-H monotonic smoothing did not converge";
      return false;
    }
    std::vector<double> next_b = curve->b_t;
    std::vector<double> next_h = curve->h_a_per_m;
    for (int i = 1; i < n - 1; ++i) {
      next_b[i] = (curve->b_t[i - 1] + curve->b_t[i] + curve->b_t[i + 1]) / 3.0;
      next_h[i] = (curve->h_a_per_m[i - 1] + curve->h_a_per_m[i]
          + curve->h_a_per_m[i + 1]) / 3.0;
    }
    curve->b_t = std::move(next_b);
    curve->h_a_per_m = std::move(next_h);
  }
  *error = "internal B-H slope error";
  return false;
}

__host__ __device__ BhEvaluation EvaluateFemmBh(double field_b,
    const double* b_t, const double* h_a_per_m, const double* slope, int count)
{
  const double b = fabs(field_b);
  BhEvaluation result = {};
  if (b == 0.0) {
    result.dh_db = slope[0];
    result.reluctivity_m_per_h = slope[0];
    return result;
  }
  if (b > b_t[count - 1]) {
    result.h_a_per_m = h_a_per_m[count - 1]
        + slope[count - 1] * (b - b_t[count - 1]);
    result.dh_db = slope[count - 1];
  } else {
    for (int i = 0; i < count - 1; ++i) {
      if (b >= b_t[i] && b <= b_t[i + 1]) {
        const double length = b_t[i + 1] - b_t[i];
        const double z = (b - b_t[i]) / length;
        const double z2 = z * z;
        result.h_a_per_m = (1.0 - 3.0 * z2 + 2.0 * z2 * z) * h_a_per_m[i]
            + z * (1.0 - 2.0 * z + z2) * length * slope[i]
            + z2 * (3.0 - 2.0 * z) * h_a_per_m[i + 1]
            + z2 * (z - 1.0) * length * slope[i + 1];
        result.dh_db = 6.0 * z * (z - 1.0) * h_a_per_m[i] / length
            + (1.0 - 4.0 * z + 3.0 * z2) * slope[i]
            + 6.0 * z * (1.0 - z) * h_a_per_m[i + 1] / length
            + z * (3.0 * z - 2.0) * slope[i + 1];
        break;
      }
    }
  }
  result.reluctivity_m_per_h = result.h_a_per_m / b;
  result.dv_db2 = 0.5 * (result.dh_db / (b * b)
      - result.h_a_per_m / (b * b * b));
  return result;
}

__global__ void EvaluateBhKernel(const double* samples, int sample_count,
    const double* b_t, const double* h_a_per_m, const double* slope,
    int point_count, BhEvaluation* output)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < sample_count)
    output[index] = EvaluateFemmBh(samples[index], b_t, h_a_per_m, slope, point_count);
}

bool ReadBhTable(const std::string& path, BhCurve* curve, std::string* error)
{
  std::ifstream input(path);
  if (!input) {
    *error = "cannot open " + path;
    return false;
  }
  std::string line;
  std::getline(input, line);
  if (line.find("H (A_per_meter)") == std::string::npos
      || line.find("B (tesla)") == std::string::npos) {
    *error = "invalid B-H table header";
    return false;
  }
  double h = 0.0;
  double b = 0.0;
  while (input >> h >> b) {
    curve->h_a_per_m.push_back(h);
    curve->b_t.push_back(b);
  }
  if (!input.eof()) {
    *error = "malformed B-H table row";
    return false;
  }
  return ComputeFemmDcSlopes(curve, error);
}

bool ReadReference(const std::string& path, std::vector<double>* samples,
    std::vector<BhEvaluation>* expected, std::string* error)
{
  std::ifstream input(path);
  if (!input) {
    *error = "cannot open " + path;
    return false;
  }
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream values(line);
    double sample = 0.0;
    BhEvaluation evaluation = {};
    if (!(values >> sample >> evaluation.h_a_per_m >> evaluation.dh_db
              >> evaluation.reluctivity_m_per_h >> evaluation.dv_db2)) {
      *error = "invalid B-H reference row";
      return false;
    }
    samples->push_back(sample);
    expected->push_back(evaluation);
  }
  if (!input.eof()) {
    *error = "malformed B-H reference";
    return false;
  }
  if (samples->empty()) {
    *error = "empty B-H reference";
    return false;
  }
  return true;
}

bool ReadSmoothingReference(const std::string& path, BhCurve* expected,
    std::string* error)
{
  std::ifstream input(path);
  if (!input) {
    *error = "cannot open " + path;
    return false;
  }
  std::string header;
  std::getline(input, header);
  if (!(input >> expected->smoothing_passes) || expected->smoothing_passes <= 0) {
    *error = "invalid smoothing pass reference";
    return false;
  }
  double b = 0.0;
  double h = 0.0;
  double slope = 0.0;
  while (input >> b >> h >> slope) {
    if (!Finite(b) || !Finite(h) || !Finite(slope)) {
      *error = "nonfinite smoothing reference";
      return false;
    }
    expected->b_t.push_back(b);
    expected->h_a_per_m.push_back(h);
    expected->slope.push_back(slope);
  }
  if (!input.eof()) {
    *error = "malformed smoothing reference";
    return false;
  }
  if (expected->b_t.size() < 2) {
    *error = "empty smoothing curve reference";
    return false;
  }
  return true;
}

bool Near(double actual, double expected, double absolute_tolerance = 1e-9,
    double relative_tolerance = 1e-12)
{
  return Finite(actual) && Finite(expected)
      && std::abs(actual - expected)
          <= absolute_tolerance + relative_tolerance * std::abs(expected);
}

template <typename T>
bool AllocateAndCopy(T** device, const std::vector<T>& host)
{
  return cudaMalloc(device, host.size() * sizeof(T)) == cudaSuccess
      && cudaMemcpy(*device, host.data(), host.size() * sizeof(T),
             cudaMemcpyHostToDevice)
          == cudaSuccess;
}

int TestFixture(const std::string& directory)
{
  BhCurve curve;
  std::vector<double> samples;
  std::vector<BhEvaluation> expected;
  BhCurve expected_smoothing;
  std::string error;
  if (!ReadBhTable(directory + "/35PN230.tab", &curve, &error)
      || !ReadReference(directory + "/reference_samples.txt", &samples,
          &expected, &error)
      || !ReadSmoothingReference(directory + "/smoothing_reference.txt",
          &expected_smoothing, &error)) {
    std::cerr << "FAIL: " << error << '\n';
    return 1;
  }
  int failures = 0;
  auto expect = [&failures](bool condition, const char* message) {
    if (!condition) {
      std::cerr << "FAIL: " << message << '\n';
      ++failures;
    }
  };
  expect(curve.b_t.size() == 29, "35PN230 point count");
  expect(curve.smoothing_passes == 0, "35PN230 FEMM smoothing-pass count");
  expect(Near(curve.slope.front(), 201.99906442618772), "first FEMM spline slope");
  expect(Near(curve.slope.back(), 793566.95194043161), "last FEMM spline slope");

  BhCurve analytic;
  analytic.b_t = { 0.0, 1.0, 2.0 };
  analytic.h_a_per_m = { 0.0, 1.0, 4.0 };
  std::string analytic_error;
  expect(ComputeFemmDcSlopes(&analytic, &analytic_error), "analytic spline construction");
  expect(analytic.slope.size() == 3 && Near(analytic.slope[0], 0.5)
          && Near(analytic.slope[1], 2.0) && Near(analytic.slope[2], 3.5),
      "analytic natural spline slopes");
  if (analytic.slope.size() == 3) {
    const BhEvaluation midpoint = EvaluateFemmBh(0.5, analytic.b_t.data(),
        analytic.h_a_per_m.data(), analytic.slope.data(), 3);
    expect(Near(midpoint.h_a_per_m, 0.3125) && Near(midpoint.dh_db, 0.875)
            && Near(midpoint.reluctivity_m_per_h, 0.625)
            && Near(midpoint.dv_db2, 0.5),
        "analytic midpoint parity");
    const BhEvaluation extrapolated = EvaluateFemmBh(3.0, analytic.b_t.data(),
        analytic.h_a_per_m.data(), analytic.slope.data(), 3);
    expect(Near(extrapolated.h_a_per_m, 7.5)
            && Near(extrapolated.dh_db, 3.5)
            && Near(extrapolated.reluctivity_m_per_h, 2.5)
            && Near(extrapolated.dv_db2, 1.0 / 18.0),
        "analytic final-slope extrapolation");
  }
  BhCurve smoothing_fixture;
  smoothing_fixture.b_t = { 0.0, 1.0, 2.0, 3.0 };
  smoothing_fixture.h_a_per_m = { 0.0, 1.0, 1.01, 10.0 };
  std::string smoothing_error;
  expect(ComputeFemmDcSlopes(&smoothing_fixture, &smoothing_error)
          && smoothing_fixture.smoothing_passes == expected_smoothing.smoothing_passes
          && smoothing_fixture.b_t.size() == expected_smoothing.b_t.size(),
      "FEMM three-point monotonic smoothing fallback");
  if (smoothing_fixture.b_t.size() == expected_smoothing.b_t.size()) {
    for (size_t i = 0; i < smoothing_fixture.b_t.size(); ++i) {
      expect(Near(smoothing_fixture.b_t[i], expected_smoothing.b_t[i])
              && Near(smoothing_fixture.h_a_per_m[i], expected_smoothing.h_a_per_m[i])
              && Near(smoothing_fixture.slope[i], expected_smoothing.slope[i]),
          "FEMM smoothed curve parity");
    }
  }

  for (size_t i = 0; i < samples.size(); ++i) {
    const BhEvaluation actual = EvaluateFemmBh(samples[i], curve.b_t.data(),
        curve.h_a_per_m.data(), curve.slope.data(),
        static_cast<int>(curve.b_t.size()));
    expect(Near(actual.h_a_per_m, expected[i].h_a_per_m), "host H parity");
    expect(Near(actual.dh_db, expected[i].dh_db), "host dH/dB parity");
    expect(Near(actual.reluctivity_m_per_h, expected[i].reluctivity_m_per_h),
        "host secant reluctivity parity");
    expect(Near(actual.dv_db2, expected[i].dv_db2), "host FEMM dv parity");
  }

  BhCurve invalid = curve;
  invalid.slope.clear();
  invalid.b_t[1] = invalid.b_t[0];
  std::string invalid_error;
  expect(!ComputeFemmDcSlopes(&invalid, &invalid_error), "duplicate B rejection");
  invalid = curve;
  invalid.slope.clear();
  invalid.h_a_per_m[1] = std::numeric_limits<double>::infinity();
  invalid_error.clear();
  expect(!ComputeFemmDcSlopes(&invalid, &invalid_error), "nonfinite H rejection");

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 1) {
    std::cerr << "FAIL: GPU_UNAVAILABLE\n";
    return 1;
  }
  double* device_samples = nullptr;
  double* device_b = nullptr;
  double* device_h = nullptr;
  double* device_slope = nullptr;
  BhEvaluation* device_output = nullptr;
  const bool allocated = AllocateAndCopy(&device_samples, samples)
      && AllocateAndCopy(&device_b, curve.b_t)
      && AllocateAndCopy(&device_h, curve.h_a_per_m)
      && AllocateAndCopy(&device_slope, curve.slope)
      && cudaMalloc(&device_output, expected.size() * sizeof(BhEvaluation)) == cudaSuccess;
  expect(allocated, "CUDA B-H buffer allocation");
  std::vector<BhEvaluation> gpu_output(expected.size());
  if (allocated) {
    const int threads = 64;
    const int blocks = (static_cast<int>(samples.size()) + threads - 1) / threads;
    EvaluateBhKernel<<<blocks, threads>>>(device_samples,
        static_cast<int>(samples.size()), device_b, device_h, device_slope,
        static_cast<int>(curve.b_t.size()), device_output);
    expect(cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess,
        "CUDA B-H kernel execution");
    expect(cudaMemcpy(gpu_output.data(), device_output,
               gpu_output.size() * sizeof(BhEvaluation), cudaMemcpyDeviceToHost)
            == cudaSuccess,
        "CUDA B-H result copy");
    for (size_t i = 0; i < expected.size(); ++i) {
      expect(Near(gpu_output[i].h_a_per_m, expected[i].h_a_per_m), "GPU H parity");
      expect(Near(gpu_output[i].dh_db, expected[i].dh_db), "GPU dH/dB parity");
      expect(Near(gpu_output[i].reluctivity_m_per_h,
                 expected[i].reluctivity_m_per_h),
          "GPU secant reluctivity parity");
      expect(Near(gpu_output[i].dv_db2, expected[i].dv_db2), "GPU FEMM dv parity");
    }
  }
  cudaFree(device_output);
  cudaFree(device_slope);
  cudaFree(device_h);
  cudaFree(device_b);
  cudaFree(device_samples);

  if (failures == 0) {
    std::cout << "PASS gpu_bh_curve_35pn230\n"
              << "  points=" << curve.b_t.size()
              << " samples=" << samples.size()
              << " smoothing_passes=" << curve.smoothing_passes << '\n'
              << "  B_max_T=" << curve.b_t.back()
              << " H_max_A_per_m=" << curve.h_a_per_m.back() << '\n';
  }
  return failures == 0 ? 0 : 1;
}

} // namespace gpu_femm

int main(int argc, char** argv)
{
  if (argc == 3 && std::string(argv[1]) == "--fixture")
    return gpu_femm::TestFixture(argv[2]);
  std::cerr << "Usage: gpu_bh_curve_poc --fixture <directory>\n";
  return 2;
}
