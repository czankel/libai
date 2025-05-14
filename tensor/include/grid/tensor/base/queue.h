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

namespace  grid::device { class Base; }
namespace grid::base {


class Queue
{
 public:
  Queue();

  /// Enqueue splits the task into blocks for the .. ranges...
  ///
  /// Enqueue({dim_x, dim_y}, [](...)
  template <typename F, typename... Args>
  void Enqueue(F&&, Args&&...);

  /// Sync synchronizes all outstanding jobs, waiting
  ///
  void Sync();

 private:
  Job current_job_;
  size_t thread_count_;
};

template <typename F, typename... Args>
void Queue::Enqueue(/*Range range,*/ F&& function, Args&&... args)
{
    //mutex;

  if (!current_job_.IsValid())
    current_job_ = worker.PostBlocked([&]() { return false; });

  size_t width = 0; // range.size();
  size_t w = ((((width + thread_count_ - 1) / thread_count_) + 7) & ~7);

  for (size_t i = 0; i < thread_count_; i++)
  {
    size3 pos{ i * w, 0, 0 };
    size3 dim{ i * w > width ? width - i * w : w, 0, 0 };
    worker.PostRunBefore(current_job_, SinusJob<double>, pos, dim, d, x);
  }
  worker.ReleaseBlocked(current_job_);
}


// FIXME: needs to take the current job and replace it??
void Queue::Sync()
{
  // FIXME: is an addiitional mutex necessary??? multiple calls to SyncThreads?? Protect current_job_?
  mutex;
  if (current_job_.IsValid())
  {
    std::condition_variable wait_cond;
    Job sync_job = worker.PostRunAfter(current_job_, [&]() -> bool {
        wait_cond.notify_all();
        return false;
        });

    if (sync.IsValid())
    {
      current_job_ = sync_job;  // FIXME

      ///std::mutex wait_mutex;
      std::unique_lock lock(wait_mutex);
      wait_cond.wait(lock);
    }
  }
}

} // end of namespace grid::base

#endif  // GRID_TENSOR_BASE_QUEUE_H
