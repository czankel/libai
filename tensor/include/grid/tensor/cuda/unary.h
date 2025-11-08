//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef LIBAI_TENSOR_CUDA_UNARY_H
#define LIBAI_TENSOR_CUDA_UNARY_H

#include <span>
#include <algorithm>
#include <ranges>

#include "device.h"
#include "../concepts.h"
#include "../unary.h"

namespace libai {

void CudaDeviceSynchronize(); // FIXME move to some header?

/// UnaryOperation<Operator> implements element-wise unary operation on a tensors.
///
///  @tparm TOperator binary operator
template <template <typename> typename TOperator>
class UnaryOperation<TOperator, device::Cuda>
{
  template <typename T, size_t R>
  void EvalContiguous(T*, const T*,
                      std::span<const size_t, R>,
                      std::span<const ssize_t, R>,
                      std::span<const ssize_t, R>) const;

  template <typename T, size_t R>
  void EvalDiscontiguous(T*, const T*,
                         std::span<const size_t, R>,
                         std::span<const ssize_t, R>,
                         std::span<const ssize_t, R>) const;

 public:

#if !defined(__CUDACC__)

  template<std::ranges::input_range I,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<I>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<I>, std::ranges::iterator_t<O>>
  void operator()(I&& in, O&& out) const
  {
    auto first_d = std::ranges::begin(out);
    auto first_x = std::ranges::cbegin(in);

    Fold([&](const auto dimensions, const auto strides_d, const auto strides_x) {

        if constexpr (dimensions.size() > 3)
          throw std::runtime_error("non-coontiguous tensors of rank > 3 not supported");

        const auto b_strides_x = BroadcastStrides<dimensions.size()>(strides_x);

        if (IsContiguous(strides_d, strides_x))
          EvalContiguous(&*first_d, &*first_x,
                         std::span(dimensions),
                         std::span(strides_d),
                         std::span(b_strides_x));
        else
          EvalDiscontiguous(&*first_d, &*first_x,
                            std::span(dimensions),
                            std::span(strides_d),
                            std::span(b_strides_x));

        CudaDeviceSynchronize(); // FIXME
    }, first_d.Extents(), first_d.Strides(), first_x.Strides());
  }

#endif  // !__CUDACC__
};


} // end of namespace libai

#endif // LIBAI_TENSOR_CUDA_UNARY_H
