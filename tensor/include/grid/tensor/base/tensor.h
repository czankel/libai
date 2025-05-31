//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_BASE_TENSOR_H
#define GRID_TENSOR_BASE_TENSOR_H

#include "array.h"

namespace grid {

template <AnyTensor T1, AnyTensor T2>
requires has_memory_type_v<T1, DeviceMemory<device::CPU>>
inline void Copy(T1& tensor1, const T2& tensor2)
{
  auto dimensions = tensor1.Dimensions();

  if (dimensions != tensor2.Dimensions())
    throw std::runtime_error("mismatching dimensions");

  details::copy_unsafe(tensor1.Data(), tensor2.Data(),
                       std::span{tensor1.Dimensions()},
                       std::span{tensor1.Strides()},
                       std::span{tensor2.Strides()});
}

} // end of namespace grid


#endif  // GRID_TENSOR_BASE_TENSOR_H
