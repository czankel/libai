//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef LIBAI_TENSOR_CUDA_RMS_NORM_H
#define LIBAI_TENSOR_CUDA_RMS_NORM_H

#include <math.h>
#include "../precision.h"

namespace libai {

void CudaDeviceSynchronize(); // FIXME move to some header?

template <> class RmsNormOperator<device::Cuda>
{
  template <typename T> void Eval(T*, const T*, const T, size_t, size_t) const;
  template <typename T> void Eval(T*, const T*, const T*, const T, size_t, size_t) const;

 public:

  // d = RMS(x) * x  -- Note: requires contiguous (not asserted)
  template<std::ranges::input_range I,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<I>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<I>, std::ranges::iterator_t<O>>
  void operator()(I&& in, O&& out) const
  {
    using value_type = std::iter_value_t<std::ranges::iterator_t<O>>;

    auto first_d = std::ranges::begin(out);
    auto first_x = std::ranges::cbegin(in);
    auto& extents = first_d.Extents();

    // FIXME: assumes contiguous??
    size_t row_size = extents.back();
    size_t n_rows = std::accumulate(std::begin(extents), std::end(extents) - 1, 1, std::multiplies<size_t>());
    Eval<value_type>(&*first_d, &*first_x, Eps<value_type>::default_value, n_rows, row_size);

    CudaDeviceSynchronize(); // FIXME
  }

  // d = RMS(x) * x @ W  -- Note: required contiguous (not asserted)
  template<std::ranges::input_range I1, std::ranges::input_range I2,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<I1>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<I1>, std::ranges::iterator_t<O>>
  void operator()(I1&& in1, I2&& in2, O&& out) const
  {
    // FIXME: implement RMS NORM with weights
#if 0
    using value_type = std::iter_value_t<std::ranges::iterator_t<O>>;

    auto first_d = std::ranges::begin(out);
    auto first_x = std::ranges::cbegin(in1);
    auto first_w = std::ranges::cbegin(in2);
    auto& extents = first_d.Extents();

    const size_t row_size = extents.back();
    const size_t n_rows = std::accumulate(std::begin(extents), std::end(extents) - 1, 1, std::multiplies<size_t>());

    if (n_rows == 1)
      Eval<value_type>(&*first_d, &*first_x, &*first_w, Eps<value_type>::value, row_size);
    else
      Eval<value_type>(&*first_d, &*first_x, &*first_w, Eps<value_type>::value, std::array<const size_t, 2>{n_rows, row_size});
    CudaDeviceSynchronize(); // FIXME
#endif
  }
};

} // end of namespace libai

#endif  // LIBAI_TENSOR_CUDA_RMS_NORM_H
