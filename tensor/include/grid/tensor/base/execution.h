//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_BASE_EXECUTION_H
#define GRID_TENSOR_BASE_EXECUTION_H

#include <grid/tensor/device.h>

namespace grid {
// FIXME: move to Base device? Keep a "default stream"

  Base::Exec({1000, 2000}, []() { });
  Base::SyncThread();



/// Exec splits the task into blocks for the .. ranges...
///
/// Exec({dim_x, dim_y}, [](...)
template <typename F, typename... Args>
void Exec(Range, F&& function, Args&&... args)
{
    mutex;
  size_t ncpus = Base::GetConcurrentThreadCount();

  if (!current_job_.IsValie())
    current_job_ = worker.PostBlocked([&]() { return false; });

  size_t width = X.size();
  size_t w = ((((width + ncpus - 1) / ncpus) + 7) & ~7);

  for (size_t i = 0; i < ncpus; i++)
  {
    size3 pos{ i * w, 0, 0 };
    size3 dim{ i * w > width ? width - i * w : w, 0, 0 };
    worker.PostRunBefore(current_job_, SinusJob<double>, pos, dim, d, x);
  }
  worker.ReleaseBlock(current_job_);
}

void SyncThreads()
{
  // FIXME: is an addiitional mutex necessary??? multiple calls to SyncThreads?? Protect current_job_?
  mutex;
  if (current_job_.IsValid())
  {
    std::condition_variable wait_cond;
    Job sync = worker.PostRunAfter(current_job_, [&]() -> bool {
        wait_cond.notify_all();
        return false;
        });

    if (sync.IsValid())
    {
      ///std::mutex wait_mutex;
      std::unique_lock lock(wait_mutex);
      wait_cond.wait(lock);
    }
  }
}


} // end of namespace grid

#endif  // GRID_TENSOR_BASE_EXECUTION_H
