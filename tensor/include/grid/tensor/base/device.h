//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_BASE_DEVICE_H
#define GRID_TENSOR_BASE_DEVICE_H

#include <grid/util/worker.h>
#include <grid/tensor/device.h>

#include "queue.h"

namespace grid::device {

class Base : public Device
{
 public:
  static inline size_t GetConcurrentThreadCount()
  {
    if (concurrent_thread_count_ == 0)
      concurrent_thread_count_ = Worker::GetConcurrentThreadCount();
    return concurrent_thread_count_;
  }

  static inline Base& GetDevice()
  {
    return g_device_;
  }

  base::Queue& GetQueue()           { return queue_; }

 private:
  static Base g_device_;
  static size_t concurrent_thread_count_;

  Job         current_job_;
  base::Queue queue_;
};

} // end of namespace grid::device

#endif  // GRID_TENSOR_BASE_DEVICE_H
