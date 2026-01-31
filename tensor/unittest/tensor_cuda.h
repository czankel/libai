//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// Important: Class template argument deduction for alias templates P1814R0 not supported on all
// compilers. This requires to duplicate *all* deduction rules in slowcpu/tensor.h

struct TensorCudaType
{
  template <typename T, size_t R>
  using Tensor = libai::Tensor<T, R, libai::device::Cuda, libai::DeviceMemory<libai::device::Cuda>>;

  template <typename T>
  using Array = libai::Array<T, libai::DeviceMemory<libai::device::Cuda>>;
};
