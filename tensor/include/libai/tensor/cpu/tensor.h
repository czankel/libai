//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef LIBAI_TENSOR_CPU_TENSOR_H
#define LIBAI_TENSOR_CPU_TENSOR_H

#include "array.h"

namespace libai {

template <AnyTensor T1, AnyTensor T2>
requires std::is_same_v<tensor_device_t<T1>, device::CPU>
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

} // end of namespace libai


#endif  // LIBAI_TENSOR_CPU_TENSOR_H
