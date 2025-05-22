//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/base/device.h>
#include <grid/tensor/base/queue.h>

using namespace grid::device;

grid::base::Queue::Queue()
{
}

// FIXME: needs to take the current job and replace it??
void grid::base::Queue::Sync()
{
  // FIXME: is an addiitional mutex necessary??? multiple calls to SyncThreads?? Protect current_job_?
  //mutex;
  if (current_job_.IsValid())
  {
    std::condition_variable wait_cond;
    Job sync_job = worker_.PostRunAfter(current_job_, [&]() -> bool {
        wait_cond.notify_all();
        return false;
        });

    if (sync_job.IsValid())
    {
      current_job_ = sync_job;  // FIXME

      std::mutex wait_mutex;
      std::unique_lock lock(wait_mutex);
      wait_cond.wait(lock);
    }
  }
}


