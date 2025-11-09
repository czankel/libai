//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef LIBAI_TENSOR_METAL_UTILS_H
#define LIBAI_TENSOR_METAL_UTILS_H

#include <algorithm>

#include <grid/util/demangle.h>

#include "device.h"
#include "kernels.h"

namespace libai {

// Helper function that aligns the dimensions to a ... FIXME
template <size_t RANK>
auto GetBlockSize(std::span<const size_t, RANK> dimensions)
{
  size_t dim0 = RANK > 0 ? dimensions[RANK - 1] : 1;
  size_t dim1 = RANK > 1 ? dimensions[RANK - 2] : 1;
  size_t rest = RANK > 2 ?
    std::accumulate(std::begin(dimensions), std::end(dimensions) - 2, 1, std::multiplies<size_t>()) : 1;

  // align dimensions to exp-2
  std::array<size_t, 3> dims = { 0, 0, 0 };
  size_t old_sum, sum = 0;
  do
  {
    old_sum = sum;
    for (size_t i = 1; i <= std::min(RANK, 3UL) && sum < 10; i++)
      if (dimensions[RANK - i] >= (2ul << dims[i - 1]))
        dims[i - 1]++, sum++;
  }
  while (sum != old_sum);

  return std::make_tuple(
      MTL::Size(dim0, dim1, rest),
      MTL::Size{1ul << dims[0], 1ul << dims[1], 1ul << dims[2]});
}

} // end of namespace libai

#endif  // LIBAI_TENSOR_METAL_UTILS_H
