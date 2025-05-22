//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_BASE_QUEUE_H
#define GRID_TENSOR_BASE_QUEUE_H

#include <grid/util/worker.h>
#include <grid/tensor/device.h>

namespace grid::base {

class Queue
{
 public:
  Queue();

  /// Enqueue splits the task into blocks of one to three dimensions
  template <size_t N, typename F, typename... Args>
  void Enqueue(size_t (&)[N], F&&, Args&&...);

  /// Sync synchronizes all outstanding jobs waiting for the jobs to complete.
  void Sync();

 private:
  Worker worker_;
  Job current_job_;
  size_t thread_count_;
};

// 8 threads 13 x X blocks --> (0,0-7), (0,8-13i & (1,0-2), ( ...
// for i = 0 to 13 * X, i += thread_count; x = (i + thread) % width, y = thread
// size_t[] tile = GetTile(thread_index, dims)
//

template <typename F, typename... Args>
bool Helper(F&& function, Args&&... args)
{
  return false;
}


// FIXME: are tiles really needed here? Number of threads should be sufficient?
template <size_t N, typename F, typename... Args>
void Queue::Enqueue(size_t (&dims)[N], F&& function, Args&&... args)
{
  Job job = current_job_;

  if (!job.IsValid())
    job = current_job_ = worker_.PostBlocked([&]() { return false; });

  //size_t width = dims[N - 1];
  //size_t w = ((((width + thread_count_ - 1) / thread_count_) + 7) & ~7);

  for (size_t i = 0; i < thread_count_; i++)
  {
    worker_.PostRunBefore(current_job_, Helper(std::forward<F>(function)));
#if 0
    //size_t pos[]{ i * w, 0, 0 };
    //size_t dim[]{ i * w > width ? width - i * w : w, 0, 0 };
    //worker_.PostRunBefore(current_job_, std::forward<F>(function), pos, dim); // FIXME , d, x);
    worker_.PostRunBefore(current_job_, [](auto f, size_t thread_id) -> bool {
        //f(pos);
        printf("i %lu\n", thread_id);
        return false;
    //}, std::forward<F>(function), i);
    }, function, i);
#endif
  }
  worker_.ReleaseBlock(current_job_);
}



} // end of namespace grid::base

#endif  // GRID_TENSOR_BASE_QUEUE_H
