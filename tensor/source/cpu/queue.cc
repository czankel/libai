//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/cpu/device.h>
#include <grid/tensor/cpu/queue.h>

using namespace libai::device;

libai::cpu::Queue::Queue() : thread_count_(libai::device::CPU::GetConcurrentThreadCount())
{
}

void libai::cpu::Queue::Sync()
{
  Job job;

  {
    std::unique_lock lock(sync_mutex_);
    job = current_job_;
    sync_ = true;
  }

  if (job.IsValid())
    job.Wait();
}
