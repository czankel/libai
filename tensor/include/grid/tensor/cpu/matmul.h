//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_CPU_MATMUL_H
#define GRID_TENSOR_CPU_MATMUL_H

#include "../device.h"

namespace grid {

/// MatmulOperator implements a multiplication operation for tensors
/// different ranks, such as matrix multiplication (Matmul) and vector dot-product (VecDot).
/// Note that all dimensions are assumed to be correct.
template <> class MatmulOperator<device::CPU>
{
  // optimized vector dot multiplication for contiguous vectors.
  template <typename T>
  inline void VecDot(T* d, const T* x, const T* y, const size_t dim) const
  {
    T sum{0};
    for (size_t n = 0; n < dim; n++)
      sum += x[n] * y[n];
    d[0] = sum;
  }

  // default vector dot multiplication for non-contigous vectors.
  template <typename T>
  inline void VecDot(T* d, const T* x, const T* y, const size_t dim,
                     const ssize_t& strides_x, const ssize_t& strides_y) const
  {
    T sum{0};
    for (size_t n = 0; n < dim; n++)
    {
      sum += x[0] * y[0];
      x += strides_x;
      y += strides_y;
    }
    d[0] = sum;
  }

  // optimized mat x vec multiplication for a contiguous matrix and vector.
  template <typename T>
  inline void MatVec(T* d, const T* x, const T* y,
                     const size_t& dim_m, const size_t& dim_n,
                     const ssize_t& strides_x) const
  {
    for (size_t m = 0; m < dim_m; m++)
    {
      T sum{0};
      for (size_t n = 0; n < dim_n; n++)
        sum += x[n] * y[n];
      d[m] = sum;
      x += strides_x;
    }
  }

  // default max x vec multiplication for non-contiguous matrix/vector.
  template <typename T>
  inline void MatVec(T* d, const T* x, const T* y,
                     const size_t& dim_m, const size_t& dim_n,
                     const ssize_t& strides_d,
                     const ssize_t& strides_x_m,
                     const ssize_t& strides_x_n,
                     const ssize_t& strides_y) const
  {
    for (size_t m = 0; m < dim_m; m++)
    {
      auto* x_prime = x;
      auto* y_prime = y;
      T sum{0};
      for (size_t n = 0; n < dim_n; n++)
      {
        sum += x_prime[0] * y_prime[0];
        x_prime += strides_x_n;
        y_prime += strides_y;
      }
      d[0] = sum;
      d += strides_d;
      x += strides_x_m;
    }
  }

  // optimized vec x mat multiplication for contiguous vector and matrix.
  template <typename T>
  inline void VecMat(T* d, const T* x, const T* y,
                     const size_t& dim_m, const size_t& dim_n,
                     const size_t& strides_n) const
  {
    for (size_t n = 0; n < dim_n; n++)
      d[n] = 0;

    for (size_t m = 0; m < dim_m; m++, y += strides_n)
      for (size_t n = 0; n < dim_n; n++)
        d[n] += x[m] * y[n];
  }

  // matrix multiplication. Note that dimensions are mn,k: M_m_k * M_k_n -> M_m_n

  // contiguous data
  template <typename T>
  inline void Matmul(T* d, const T* x, const T* y,
                     std::span<const size_t,  2> dimensions, size_t dim_k) const
  {
    for (size_t m = 0; m < dimensions[0]; m++)
    {
      const T* y_prime = y;
      for (size_t n = 0; n < dimensions[1]; n++)
      {
        T sum{0};
        for (size_t k = 0; k < dim_k; k++)
          sum += x[k] * y_prime[k];
        d[n] = sum;
        y_prime += dim_k;
      }
      x += dim_k;
      d += dimensions[1];
    }
  }

  // semi-optimized: only lowest 'rank' is contiguous and rhs transposed
  template <typename T>
  inline void Matmul(T* d, const T* x, const T* y,
                     std::span<const size_t,  2> dimensions, size_t dim_k,
                     const ssize_t& strides_d, const ssize_t& strides_x, const ssize_t& strides_y) const
  {
    for (size_t m = 0; m < dimensions[0]; m++)
    {
      T* d_prime = d;
      const T* y_prime = y;
      for (size_t n = 0; n < dimensions[1]; n++)
      {
        T sum{0};
        for (size_t k = 0; k < dim_k; k++)
          sum += x[k] * y_prime[k];
        d_prime[n] = sum;
        y_prime += strides_y;
      }
      d += strides_d;
      x += strides_x;
    }
  }

  // default unoptimized matrix multiplication
  template <typename T>
  inline void Matmul(T* d, const T* x, const T* y,
                     std::span<const size_t,  2> dimensions,
                     size_t                      dim_k,
                     std::span<const ssize_t, 2> strides_d,
                     std::span<const ssize_t, 2> strides_x,
                     std::span<const ssize_t, 2> strides_y) const
  {
    for (size_t m = 0; m < dimensions[0]; m++)
    {
      T* d_prime = d;
      const T* y_prime = y;
      for (size_t n = 0; n < dimensions[1]; n++)
      {
        const T* x_tmp = x;
        const T* y_tmp = y_prime;
        T sum{0};
        for (size_t i = 0; i < dim_k; i++)
        {
          sum += *x_tmp * *y_tmp;
          x_tmp += strides_x[1];
          y_tmp += strides_y[0];
        }
        *d_prime = sum;
        d_prime += strides_d[1];
        y_prime += strides_y[1];
      }
      d += strides_d[0];
      x += strides_x[0];
    }
  }

 public:
  template<std::ranges::input_range I1,
           std::ranges::input_range I2,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<I1>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<I1>, std::ranges::iterator_t<O>> &&
           std::indirectly_copyable<std::ranges::iterator_t<I2>, std::ranges::iterator_t<O>>
  void operator()(I1&& in1, I2&& in2, O&& out) const
  {
    auto first_d = std::ranges::begin(out);
    auto first_x = std::ranges::cbegin(in1);
    auto first_y = std::ranges::cbegin(in2);

    constexpr size_t rank_x = std::ranges::iterator_t<I1>::rank;
    constexpr size_t rank_y = std::ranges::iterator_t<I2>::rank;

    auto& strides_d = first_d.Strides();
    auto& strides_x = first_x.Strides();
    auto& strides_y = first_y.Strides();

    // FIXME: use std::moves for strides, dims, etc?

    // mat * mat: M_m_k * M_k_n -> M_m_n
    if constexpr (rank_x == 2 && rank_y == 2)
    {
      size_t dim_k = first_x.Extents()[1];
      auto& extents = first_d.Extents();
      if (strides_d[1] <= 1 && strides_x[1] <= 1 && strides_y[0] <= 1)
      {
        // full optimizations: mat * mat and all tensors are contiguous, strides ignored
        if (strides_d[0] - extents[1] == 0 &&
            strides_x[0] - dim_k == 0 &&
            strides_y[1] - dim_k == 0)
          Matmul(&*first_d, &*first_x, &*first_y, std::span(extents), dim_k);

        // semi-contiguous
        else
        {
          Matmul(&*first_d, &*first_x, &*first_y, std::span(extents), dim_k,
                 strides_d[0], strides_x[0], strides_y[1]); }
      }
      else
        Matmul(&*first_d, &*first_x, &*first_y, std::span(extents), dim_k,
               std::span(strides_d), std::span(strides_x), std::span(strides_y));
    }

    // mat * vec: M_m_n * V_n = M_m_n * V_n_1 -> V_m_1 = V_m
    else if constexpr (rank_x == 2 && rank_y == 1)
    {
      auto& extents = first_x.Extents();
      if (strides_d[0] <= 1 && strides_x[1] <= 1 && strides_y[0] == 1)
        MatVec(&*first_d, &*first_x, &*first_y, extents[0], extents[1], strides_x[0]);
      else
        MatVec(&*first_d, &*first_x, &*first_y, extents[0], extents[1],
               strides_d[0], strides_x[0], strides_x[1], strides_y[0]);
    }

    // vec * mat: V_m * M_m_n = V_1_m * M_m_n -> V_1_n = V_n (note: pass transposed dims/strides)
    else if constexpr (rank_x == 1 && rank_y == 2)
    {
      auto& extents = first_y.Extents();
      if (strides_d[0] == 1 && strides_x[0] == 1 && strides_y[1] == 1)
        VecMat(&*first_d, &*first_x, &*first_y, extents[0], extents[1], strides_y[0]);
      else if (strides_d[0] == 1 && strides_x[0] == 1 && strides_y[0] == 1)
        MatVec(&*first_d, &*first_y, &*first_x, extents[1], extents[0], strides_y[1]);
      else
        MatVec(&*first_d, &*first_y, &*first_x, extents[1], extents[0],
               strides_d[0], strides_y[1], strides_y[0], strides_x[0]);
    }

    // vecdot: V_m * V_m -> scalar
    else if constexpr (rank_x == 1 && rank_y == 1)
    {
      if (strides_x[0] == 1 && strides_y[0] == 1)
        VecDot(&*first_d, &*first_x, &*first_y, first_x.Extents()[0]);
      else
        VecDot(&*first_d, &*first_x, &*first_y, first_x.Extents()[0], strides_x[0], strides_y[0]);
    }
  }
};

} // end of namespace grid

#endif  // GRID_TENSOR_CPU_MATMUL_H
