//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_CPU_RMS_NORM_H
#define GRID_TENSOR_CPU_RMS_NORM_H

#include <math.h>
#include <tuple>
#include <iomanip>

#include "binary.h"
#include "../precision.h"

namespace grid {

// requires (std::is_floating_point_v<value_type> && rank > 0)
template <> class RmsNormOperator<device::CPU>
{
 private:

  template <typename T>
  inline auto
  SumSquare(const T* x, const size_t dim, const ssize_t stride) const
  {
    T value{0};
    for (size_t i = 0; i < dim; i++, x += stride)
      value += *x * *x;
    return value;
  }

 public:
  template<std::ranges::input_range I, std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<I>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<I>, std::ranges::iterator_t<O>>
  void operator()(I&& in, O&& out) const
  {
    using tensor_type = std::remove_cvref_t<O>;
    using value_type = tensor_type::value_type;
    constexpr value_type eps = Eps<value_type>::default_value;
    constexpr size_t rank = tensor_type::rank;

    auto first_d = std::ranges::begin(out);
    auto first_x = std::ranges::cbegin(in);

    size_t stride_x = first_x.Strides().back();
    auto& extents = first_d.Extents();
    size_t row_size = extents.back();

    if constexpr (rank == 1)
    {
      auto sum = SumSquare(&*first_x, row_size, stride_x);
      value_type scale = sqrtf(sum / row_size + eps);
      BinaryOperation<DivOperator, device::CPU>()(in, Tensor(scale), out);
    }
    else
    {
      size_t n_rows =
        std::accumulate(std::begin(extents), std::end(extents) - 1, 1, std::multiplies<size_t>());

      // TODO could there be an alias issue?
      auto x = &*first_x;
      Tensor scale({n_rows, 1}, Uninitialized<value_type>{});
      for (size_t row = 0; row < n_rows; row++)
      {
        auto sum = SumSquare(x, row_size, stride_x);
        scale.Data()[row] = sqrtf(sum / row_size + eps);
        x += first_x.Strides()[rank - 2];
      }
      BinaryOperation<DivOperator, device::CPU>()(in, scale, out);
    }
  }
};

} // end of namespace grid

#endif  // GRID_TENSOR_CPU_RMS_NORM_H
