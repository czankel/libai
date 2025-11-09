//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef LIBAI_TENSOR_CPU_QUEUE_H
#define LIBAI_TENSOR_CPU_QUEUE_H

#include <numeric>

#include <grid/util/worker.h>
#include <grid/tensor/device.h>

namespace libai::cpu {


class Queue
{
 public:
  Queue();

  /// Enqueue splits the task into blocks of one to three dimensions
  ///
  /// The sizes defines the dimensions of each block. The caller should align these
  /// to the cache line size; Enque uses them as is.
  ///
  /// @param dims  Dimensions of the entire array
  /// @param sizes Dimensions of a block
  template <size_t N, typename F, typename... Args>
  void Enqueue(const std::array<size_t, N>& dims, const std::array<size_t, N>& sizes, F&& f, Args&&... args);

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
void Queue::Enqueue(const std::array<size_t, N>& dims, const std::array<size_t, N>& sizes, F&& function, Args&&... args)
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
    else
      current_job_.GetWorker().Block(current_job_);
  }

  size_t dim = std::accumulate(std::begin(dims), std::end(dims), 1, std::multiplies<size_t>());
  size_t size = std::accumulate(std::begin(sizes), std::end(sizes), 1, std::multiplies<size_t>());
  size_t n_threads = (dim + size - 1) / size;
  if (n_threads > thread_count_)
    n_threads = thread_count_;

  for (size_t thread_id = 0; thread_id < n_threads; thread_id++)
  {
    worker_.PostRunBefore(current_job_, [n_threads, dims, sizes, args...](auto f, size_t thread_id) -> bool
      {
        if constexpr (N == 1)
        {
          size_t n_blocks = (dims[0] + sizes[0] - 1) / sizes[0];
          for (size_t pos = thread_id; pos < n_blocks; pos += n_threads)
            f(std::to_array<size_t>({pos}), dims, sizes, args...);
        }
        else if constexpr (N == 2)
        {
          size_t n_rows = (dims[0] + sizes[0] - 1) / sizes[0];
          size_t n_cols = (dims[1] + sizes[1] - 1) / sizes[1];
          for (size_t pos = thread_id; pos < n_rows * n_cols; pos += n_threads)
            f(std::to_array<size_t>({pos / n_cols, pos % n_cols}), dims, sizes, args...);
        }
        else if constexpr (N == 3)
        {
        // FIXME: channel or batch instead of depth?
          size_t n_depths = (dims[0] + sizes[0] - 1) / sizes[0];
          size_t n_rows = (dims[1] + sizes[1] - 1) / sizes[1];
          size_t n_cols = (dims[2] + sizes[2] - 1) / sizes[2];
          for (size_t pos = thread_id; pos < n_depths * n_rows * n_cols; pos += n_threads)
            f(std::to_array<size_t>({pos / n_cols / n_rows, (pos / n_cols) % n_rows, pos % n_cols}), dims, sizes, args...);
        }
        return false;
    }, std::forward<F>(function), thread_id);
  }
  worker_.ReleaseBlock(current_job_);
}

} // end of namespace libai::cpu

#endif  // LIBAI_TENSOR_CPU_QUEUE_H
