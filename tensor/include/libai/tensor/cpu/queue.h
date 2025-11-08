//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef LIBAI_TENSOR_CPU_QUEUE_H
#define LIBAI_TENSOR_CPU_QUEUE_H

#include <libai/util/worker.h>
#include <libai/tensor/device.h>

namespace libai::cpu {


class Queue
{
 public:
  Queue();

  /// Enqueue splits the task into blocks of one to three dimensions
  template <size_t N, typename F, typename... Args> requires (N > 0 && N < 4)
  void Enqueue(size_t (&)[N], size_t (&)[N], F&&, Args&&...);

  /// Sync synchronizes all outstanding jobs waiting for the jobs to complete.
  void Sync();

 private:
  Worker  worker_;
  size_t  thread_count_;

  std::mutex  sync_mutex_;
  Job         current_job_;
  bool        sync_;
};


template <size_t N, typename F, typename... Args>
requires (N > 0 && N < 4)
void Queue::Enqueue(size_t (&dims)[N], size_t (&sizes)[N], F&& function, Args&&... args)
{
  {
    std::unique_lock lock(sync_mutex_);
    if (!current_job_.IsValid() || sync_)
    {
      sync_ = false;
      current_job_ = worker_.PostBlocked([&]() {
          std::unique_lock lock(sync_mutex_);
          return !sync_ || (current_job_ == CurrentJob::GetJob()); });
    }
  }

  size_t size = (dims[N-1] * (N > 1? dims[N-2] : 1) * (N > 2? dims[N-3] : 1) + sizes[N-1] - 1) / sizes[N-1];
  size_t n_threads = size > thread_count_ ? thread_count_ : size;

  for (size_t thread_id = 0; thread_id < n_threads; thread_id++)
  {
    worker_.PostRunBefore(current_job_, [n_threads, &dims, &sizes, args...](auto f, size_t thread_id) -> bool {

        if constexpr (N == 1)
        {
          for (size_t pos = thread_id; pos < (dims[0] + sizes[0] - 1) / sizes[0]; pos += n_threads)
              f({pos}, args...);
        }
        else if constexpr (N == 2)
        {
          size_t n_rows = (dims[0] + sizes[0] - 1) / sizes[0];
          size_t n_cols = (dims[1] + sizes[1] - 1) / sizes[1];
          for (size_t pos = thread_id; pos < n_rows * n_cols; pos += n_threads)
            f({pos / n_cols, pos % n_cols}, args...);
        }
        else if constexpr (N == 3)
        {
          size_t n_depths = (dims[0] + sizes[0] - 1) / sizes[0];
          size_t n_rows = (dims[1] + sizes[1] - 1) / sizes[1];
          size_t n_cols = (dims[2] + sizes[2] - 1) / sizes[2];
          for (size_t pos = thread_id; pos < n_depths * n_rows * n_cols; pos += n_threads)
            f({pos / n_cols / n_rows, (pos / n_cols) % n_rows, pos % n_cols}, args...);
        }
        return false;
    }, std::forward<F>(function), thread_id);
  }
  worker_.ReleaseBlock(current_job_);
}

} // end of namespace libai::cpu

#endif  // LIBAI_TENSOR_CPU_QUEUE_H
