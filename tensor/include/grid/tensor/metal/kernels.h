//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_METAL_KERNEL_H
#define GRID_TENSOR_METAL_KERNEL_H

#include <stdio.h>
#include <grid/util/demangle.h>

#include "device.h"

namespace libai {
namespace metal {

template <typename T>
class Kernel
{
 public:
  Kernel(std::string name)
  {
    auto& dev = device::Metal::GetDevice();
    auto tp = libai::Demangle(typeid(T).name());
    if (tp == "int")
      tp = "int32_t";
    pipeline_ = dev.GetKernel(name + tp);
  }

  MTL::ComputePipelineState* ComputePipelineState()       { return pipeline_; }

 private:
  MTL::ComputePipelineState* pipeline_;
};

} // end of namespace metal
} // end of namespace libai

#endif  // GRID_TENSOR_METAL_KERNEL_H
