#include "app/motor_worker.h"
#include "app/motor_batch.h"
#include "app/sample_adapter.h"
#include "core/profile.h"
#include "io/json_output.h"
#include "io/strict_json.h"
#include "solvers/options.h"

#include <condition_variable>
#include <algorithm>
#include <atomic>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace gpu_femm
{
namespace
{
struct Job {
  std::string session, request_path, response_path, request_sha256, error;
  int32_t sequence = 0;
  ProfileClock::time_point prepare_begin, prepare_end;
  std::unique_ptr<MotorBatchPrefetch> input;
};

std::unique_ptr<Job> PrepareJob(const std::string &line, const char *executable)
{
  auto job = std::make_unique<Job>();
  const auto started = ProfileClock::now();
  job->prepare_begin = started;
  try {
    StrictJson root;
    std::string error;
    if (!StrictJsonParser(line).Parse(&root, &error) ||
        !ExactObject(root, {"session", "sequence", "request_path", "response_path",
                            "request_sha256"}, &error))
      throw std::runtime_error("invalid stream job envelope");
    const auto *session = JsonMember(root, "session", StrictJson::Type::kString, &error);
    const auto *sequence = JsonMember(root, "sequence", StrictJson::Type::kNumber, &error);
    const auto *request = JsonMember(root, "request_path", StrictJson::Type::kString, &error);
    const auto *response = JsonMember(root, "response_path", StrictJson::Type::kString, &error);
    const auto *hash = JsonMember(root, "request_sha256", StrictJson::Type::kString, &error);
    if (!session || !sequence || !request || !response || !hash ||
        !JsonLowerSha256(session->string) || !JsonLowerSha256(hash->string) ||
        !JsonInteger(*sequence, &job->sequence) || job->sequence < 1)
      throw std::runtime_error("invalid stream job identity");
    job->session = session->string;
    job->request_path = request->string;
    job->response_path = response->string;
    job->request_sha256 = hash->string;
    if (job->request_path.empty() || job->response_path.empty() ||
        SameFilesystemObject(job->response_path, job->request_path) ||
        SameFilesystemObject(job->response_path, executable))
      throw std::runtime_error("unsafe stream output path");
    std::ifstream file(job->request_path, std::ios::binary);
    std::stringstream bytes;
    bytes << file.rdbuf();
    if (!file || Sha256Hex(bytes.str()) != job->request_sha256)
      throw std::runtime_error("stream request SHA-256 mismatch");
    auto input = std::make_unique<MotorBatchPrefetch>();
    if (!ReadMotorBatchRequestJson(bytes.str(), &input->request, &error))
      throw std::runtime_error(error);
    // A transaction is bounded independently of the number of submissions.
    // The MATLAB producer splits larger requests using the same wire cap.
    if (input->request.items.size() > 32)
      throw std::runtime_error("stream transaction exceeds 32 items");
    input->load_plan = MakeMotorBatchArtifactLoadPlan(input->request);
    for (const auto &path : input->load_plan.representative_paths) {
      if (SameFilesystemObject(job->response_path, path))
        throw std::runtime_error("stream response would overwrite an artifact");
      input->loads.push_back(std::make_unique<MotorBatchArtifactLoad>());
    }
    std::atomic<size_t> next_slot{0};
    auto load = [&] {
      for (;;) {
        const size_t slot = next_slot.fetch_add(1, std::memory_order_relaxed);
        if (slot >= input->loads.size())
          return;
        LoadMotorBatchArtifact(input->load_plan.representative_paths[slot],
                              input->loads[slot].get());
      }
    };
    std::vector<std::thread> loaders;
    const size_t width = std::min<size_t>(4, input->loads.size());
    for (size_t i = 1; i < width; ++i) {
      try {
        loaders.emplace_back(load);
      } catch (const std::system_error &) {
        break;
      }
    }
    load();
    for (auto &thread : loaders)
      thread.join();
    input->prepare_seconds = ProfileSecondsSince(started);
    job->input = std::move(input);
  } catch (const std::exception &error) {
    job->error = error.what();
  }
  job->prepare_end = ProfileClock::now();
  return job;
}
}

int MotorBatchWorker(const char *executable_path)
{
  const auto worker_start = ProfileClock::now();
  std::mutex mutex;
  std::condition_variable changed;
  std::unique_ptr<Job> ready;
  bool eof = false;
  std::thread producer([&] {
    for (;;) {
      // Wait BEFORE reading/parsing: at most one prepared input is retained
      // in addition to the transaction currently using the GPU.
      std::unique_lock<std::mutex> lock(mutex);
      changed.wait(lock, [&] { return !ready; });
      lock.unlock();
      std::string line;
      if (!std::getline(std::cin, line))
        break;
      auto job = PrepareJob(line, executable_path);
      lock.lock();
      ready = std::move(job);
      changed.notify_all();
    }
    std::lock_guard<std::mutex> lock(mutex);
    eof = true;
    changed.notify_all();
  });
  std::string session;
  int32_t previous_sequence = 0;
  for (;;) {
    std::unique_lock<std::mutex> lock(mutex);
    changed.wait(lock, [&] { return ready || eof; });
    if (!ready)
      break;
    auto job = std::move(ready);
    changed.notify_all();
    lock.unlock();
    int result = 1;
    const auto solve_begin = ProfileClock::now();
    const double prepare_seconds = job->input ? job->input->prepare_seconds : 0.0;
    if (job->error.empty()) {
      if ((session.empty() || session == job->session) &&
          job->sequence == previous_sequence + 1) {
        session = job->session;
        previous_sequence = job->sequence;
        try {
          // Per-transaction diagnostics must not inherit earlier job counts.
          direct_solver_stats = {};
          result = MotorBatchAdapter(job->request_path, job->response_path, true,
                                    NonlinearAssemblyMode::kAuto, std::move(job->input));
        } catch (const std::exception &error) {
          job->error = error.what();
        }
      } else {
        job->error = "stream session or submission sequence mismatch";
      }
    } else if ((session.empty() || session == job->session) &&
               job->sequence == previous_sequence + 1) {
      // A rejected request still consumes its envelope number, not GPU work.
      session = job->session;
      previous_sequence = job->sequence;
    }
    const auto solve_end = ProfileClock::now();
    const auto since_start = [&](const ProfileClock::time_point &point) {
      return std::chrono::duration<double>(point - worker_start).count();
    };
    std::cout << "{\"session\":\"" << EscapeJsonString(job->session)
              << "\",\"sequence\":" << job->sequence
              << ",\"request_sha256\":\"" << job->request_sha256
              << "\",\"exit_code\":" << result
              << ",\"prepare_seconds\":" << prepare_seconds
              << ",\"prepare_begin_seconds\":" << since_start(job->prepare_begin)
              << ",\"prepare_end_seconds\":" << since_start(job->prepare_end)
              << ",\"solve_begin_seconds\":" << since_start(solve_begin)
              << ",\"solve_end_seconds\":" << since_start(solve_end)
              << ",\"error\":\"" << EscapeJsonString(job->error) << "\"}" << std::endl;
  }
  producer.join();
  return 0;
}
} // namespace gpu_femm
