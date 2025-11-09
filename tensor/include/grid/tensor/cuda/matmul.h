//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_CUDA_MATMUL_H
#define GRID_TENSOR_CUDA_MATMUL_H


#include "grid/util/demangle.h"

#include "device.h"

namespace libai {

void CudaDeviceSynchronize(); // FIXME move to some header?

/// MatmulOperator implements a multiplication operation for tensors
/// different ranks, such as matrix multiplication (Matmul) and vector dot-product (VecDot).
template <> class MatmulOperator<device::Cuda>
{
  template <typename T>
  void EvalGeneric(T*, const T*, const T*,
                   std::span<const size_t, 3>,
                   std::span<const ssize_t, 2>, std::span<const ssize_t, 2>, std::span<const ssize_t, 2>) const;

public:

#if !defined(__CUDACC__)

  // FIXME: will fold work for matmul?
  template<std::ranges::input_range I1,
           std::ranges::input_range I2,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<I1>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<I1>, std::ranges::iterator_t<O>> &&
           std::indirectly_copyable<std::ranges::iterator_t<I2>, std::ranges::iterator_t<O>>
  void operator()(I1&& in1, I2&& in2, O&& out) const
  {
    using value_type = std::iter_value_t<std::ranges::iterator_t<O>>;

    auto first_d = std::ranges::begin(out);
    auto first_x = std::ranges::cbegin(in1);
    auto first_y = std::ranges::cbegin(in2);

    constexpr size_t rank_x = std::ranges::iterator_t<I1>::rank;
    constexpr size_t rank_y = std::ranges::iterator_t<I2>::rank;

    // FIXME: vecdot has rank 0!!!
    size_t dim_j = first_x.Extents()[rank_x - 1];
    auto& dims_x = first_x.Extents();
    auto& dims_y = first_y.Extents();

    if (dim_j != first_y.Extents()[0])
      throw std::runtime_error("GeMM: inner dimensions don't match");

    if constexpr (rank_x == 2 && rank_y == 2)
      EvalGeneric<value_type>(&*first_d, &*first_x, &*first_y,
                              std::array<size_t, 3>{dims_x[0], dim_j, dims_y[1]},
                              std::span(first_d.Strides()),
                              std::span(first_x.Strides()),
                              std::span(first_y.Strides()));
    else if constexpr (rank_x == 2 && rank_y == 1)
      EvalGeneric<value_type>(&*first_d, &*first_x, &*first_y,
                              std::array<size_t, 3>{dims_x[0], dim_j, 1},
                              std::array<const ssize_t, 2>{first_d.Strides()[0], 0},
                              std::span(first_x.Strides()),
                              std::array<const ssize_t, 2>{first_y.Strides()[0], 0});
    else if constexpr (rank_x == 1 && rank_y == 2)
      EvalGeneric<value_type>(&*first_d, &*first_x, &*first_y,
                              std::array<size_t, 3>{1, dim_j, dims_y[1]},
                              std::array<const ssize_t, 2>{0, first_d.Strides()[0]},
                              std::array<const ssize_t, 2>{0, first_x.Strides()[0]},
                              std::span(first_y.Strides()));
    else if constexpr (rank_x == 1 && rank_y == 1)
      EvalGeneric<value_type>(&*first_d, &*first_x, &*first_y,
                              std::array<size_t, 3>{1, dim_j, 1},
                              std::array<const ssize_t, 2>{0, 0},
                              std::array<const ssize_t, 2>{0, first_x.Strides()[0]},
                              std::array<const ssize_t, 2>{first_y.Strides()[0], 0});
    else
      throw std::runtime_error("invalid matrix/vector multiplication");

    CudaDeviceSynchronize(); // FIXME
  }


#endif  // !__CUDACC__
};

} // end of namespace libai

#endif  // GRID_TENSOR_CUDA_MATMUL_H
