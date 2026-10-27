#pragma once

namespace gpu_femm
{
// JSON-lines transport on stdin/stdout. One CPU producer preloads at most
// one following transaction while the calling thread alone owns CUDA.
int MotorBatchWorker(const char *executable_path);
}
