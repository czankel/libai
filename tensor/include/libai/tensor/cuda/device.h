//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef LIBAI_TENSOR_CUDA_DEVICE_H
#define LIBAI_TENSOR_CUDA_DEVICE_H

#include "allocator.h"
#include "../device.h"

namespace libai::device {

class Cuda : public Device
{
  Cuda();
  Cuda(Cuda&) = delete;
  Cuda& operator =(Cuda&) = delete;

 public:
  template <typename T> using allocator_type = CudaAllocator<T>;

  ~Cuda();

  /// @brief Returns the default devices (using a singleton)
  static Cuda& GetDevice();

  /// @brief Returns the number of devices in the system
  static int GetDeviceCount();

 private:
  static Cuda*      g_device_;
};

} // end of namespace libai::device

#endif  // LIBAI_TENSOR_CUDA_DEVICE_H
