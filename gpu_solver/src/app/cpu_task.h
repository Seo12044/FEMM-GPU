#pragma once

#include <future>
#include <memory>
#include <system_error>
#include <type_traits>
#include <utility>

namespace gpu_femm
{
// Keep move-owned inputs intact even if the OS refuses a worker thread.
template <class Function> auto LaunchCpuTask(Function &&function)
{
  auto work = std::make_shared<std::decay_t<Function>>(std::forward<Function>(function));
  auto invoke = [work]() mutable { return (*work)(); };
  try {
    return std::async(std::launch::async, invoke);
  } catch (const std::system_error &) {
    return std::async(std::launch::deferred, invoke);
  }
}
} // namespace gpu_femm
