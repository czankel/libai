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

namespace libai {

/// MatmulOperator implements a multiplication operation for tensors
/// different ranks, such as matrix multiplication (Matmul) and vector dot-product (VecDot).
/// Note that all dimensions are assumed to be correct.
template <> class MatmulOperator<device::CPU>
{
  // optimized mat x vec multiplication for a contiguous matrix and vector.
  template <typename T>
  static inline void MatVec(T* d, const T* x, const T* y,
                            std::span<const size_t, 2> dims,
                            const ssize_t& strides_x)
  {
    for (size_t m = 0; m < dims[0]; m++)
    {
      T sum{0};
      for (size_t n = 0; n < dims[1]; n++)
        sum += x[n] * y[n];
      d[m] = sum;
      x += strides_x;
    }
  }

  // default max x vec multiplication for non-contiguous matrix/vector.
  template <typename T>
  static inline void MatVec(T* d, const T* x, const T* y,
                            std::span<const size_t, 2> dims,
                            const ssize_t& strides_d,
                            const ssize_t& strides_x_m,
                            const ssize_t& strides_x_n,
                            const ssize_t& strides_y)
  {
    for (size_t m = 0; m < dims[0]; m++)
    {
      auto* x_prime = x;
      auto* y_prime = y;
      T sum{0};
      for (size_t n = 0; n < dims[1]; n++)
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
  static inline void VecMat(std::span<const size_t, 2> dims,
                            T* d, const T* x, const T* y,
                            const size_t& strides_n)
  {
    for (size_t n = 0; n < dims[1]; n++)
      d[n] = 0;

    for (size_t m = 0; m < dims[0]; m++, y += strides_n)
      for (size_t n = 0; n < dims[1]; n++)
        d[n] += x[m] * y[n];
  }

  // matrix multiplication. Note that dimensions are mn,k: M_m_k * M_k_n -> M_m_n

  // fully optimized: contiguous data; MxN N is stored transposed
  template <typename T>
  static inline void Matmul(std::span<const size_t, 2> pos,
                            std::span<const size_t, 2> dims,
                            std::span<const size_t, 2> sizes,
                            T* d, const T* x, const T* y,
                            size_t dim_k)
  {
    size_t offset_m = pos[0] * sizes[0]; 
    size_t offset_n = pos[1] * sizes[1];

    d += offset_m * dims[1] + offset_n;
    x += offset_m * dim_k;
    y += offset_n;

    for (size_t m = offset_m; m < offset_m + sizes[0] && m < dims[0]; m++)
    {
      const T* y_prime = y;
      for (size_t n = offset_n; n < offset_n + sizes[1] && n < dims[1]; n++)
      {
        T sum{0};
        for (size_t k = 0; k < dim_k; k++)
          sum += x[k] * y_prime[k];
        d[n] = sum;
        y_prime += dim_k;
      }
      x += dim_k;
      d += dims[1];
    }
  }

  // semi-optimized: only lowest 'rank' is contiguous and rhs transposed
  template <typename T>
  static inline void Matmul(std::span<const size_t, 2> pos,
                            std::span<const size_t, 2> dimensions,
                            std::span<const size_t, 2> sizes,
                            T* d, const T* x, const T* y,
                            size_t dim_k,
                            const ssize_t strides_d,
                            const ssize_t strides_x,
                            const ssize_t strides_y)
  {
    size_t offset_m = pos[0] * sizes[0];
    size_t offset_n = pos[1] * sizes[1];

    d += offset_m * strides_d;
    x += offset_m * strides_x;
    y += offset_n * strides_y;

    for (size_t m = offset_m; m < offset_m + sizes[0] && m < dimensions[0]; m++)
    {
      T* d_prime = d;
      const T* y_prime = y;

      for (size_t n = offset_n; n < sizes[1] && n < dimensions[1]; n++)
      {
        T sum{0};
        for (size_t k = 0; k < dim_k; k++)
          sum += x[k] * y_prime[k];
        d_prime[n] = sum; // FIXME: isn't this just d[n]?
        y_prime += strides_y;
      }
      d += strides_d;
      x += strides_x;
    }
  }

  // default unoptimized matrix multiplication
  template <typename T>
  static inline void Matmul(std::span<const size_t, 2> pos,
                            std::span<const size_t, 2> dimensions,
                            std::span<const size_t, 2> sizes,
                            T* d, const T* x, const T* y,
                            size_t dim_k,
                            std::span<const ssize_t, 2> strides_d,
                            std::span<const ssize_t, 2> strides_x,
                            std::span<const ssize_t, 2> strides_y)
  {
    size_t offset_m = pos[0] * sizes[0];
    size_t offset_n = pos[1] * sizes[1];

    d += offset_m * strides_d[0];
    x += offset_m * strides_x[0];
    y += offset_n * strides_y[1];

    for (size_t m = 0; m < sizes[0] && m < dimensions[0]; m++)
    {
      T* d_prime = d;
      const T* y_prime = y;
      for (size_t n = 0; n < sizes[1] && n < dimensions[1]; n++)
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


  template <typename T, size_t RankD, size_t RankX, size_t RankY>
  static inline void Eval(std::span<const size_t, RankD> pos,
                          std::span<const size_t, RankD> dimensions,
                          std::span<const size_t, RankD> sizes,
                          std::span<const size_t, RankX> extents_x,
                          std::span<const size_t, RankY> extents_y,
                          T* d, const T* x, const T* y,
                          std::span<const ssize_t, RankD> strides_d,
                          std::span<const ssize_t, RankX> strides_x,
                          std::span<const ssize_t, RankY> strides_y)
  {
    // mat * mat: M_m_k * M_k_n -> M_m_n
    if constexpr (RankX == 2 && RankY == 2)
    {
      size_t dim_k = extents_x[1];
      if (strides_d[1] <= 1 && strides_x[1] <= 1 && strides_y[0] <= 1)
      {
        // full optimizations: mat * mat and all tensors are contiguous, strides ignored
        if (strides_d[0] - dimensions[1] == 0 &&
            strides_x[0] - dim_k == 0 &&
            strides_y[1] - dim_k == 0)
          Matmul(std::move(pos), std::move(dimensions), std::move(sizes),
                 d, x, y, dim_k);
        // semi-contiguous
        else
        {
          Matmul(std::move(pos), std::move(dimensions), std::move(sizes),
                 d, x, y, dim_k,
                 strides_d[0], strides_x[0], strides_y[1]); }
      }
      else
        Matmul(std::move(pos), std::move(dimensions), std::move(sizes),
               d, x, y, dim_k,
               std::span(strides_d), std::span(strides_x), std::span(strides_y));
    }

    // mat * vec: M_m_n * V_n = M_m_n * V_n_1 -> V_m_1 = V_m
    else if constexpr (RankX == 2 && RankY == 1)
    {
      if (strides_d[0] <= 1 && strides_x[1] <= 1 && strides_y[0] == 1)
        MatVec(d, x, y, extents_x, strides_x[0]);
      else
        MatVec(d, x, y, extents_x,
               strides_d[0], strides_x[0], strides_x[1], strides_y[0]);
    }

    // vec * mat: V_m * M_m_n = V_1_m * M_m_n -> V_1_n = V_n (note: pass transposed dims/strides)
    else if constexpr (RankX == 1 && RankY == 2)
    {
      if (strides_d[0] == 1 && strides_x[0] == 1 && strides_y[1] == 1)
        VecMat(extents_y, d, x, y, strides_y[0]);
      else if (strides_d[0] == 1 && strides_x[0] == 1 && strides_y[0] == 1)
        MatVec(d, y, x,
               std::move(get_array({extents_y[1], extents_y[0]})), strides_y[1]);
      else
        MatVec(d, y, x,
               std::move(get_array({extents_y[1], extents_y[0]})),
               strides_d[0], strides_y[1], strides_y[0], strides_x[0]);
    }

    // vecdot: V_m * V_m -> scalar
    else if constexpr (RankX == 1 && RankY == 1)
    {
      T sum{0};
      if (strides_x[0] == 1 && strides_y[0] == 1)
      {
        for (size_t n = 0; n < extents_x[0]; n++)
          sum += x[n] * y[n];
      }
      else
      {
        for (size_t n = 0; n < extents_x[0]; n++)
        {
          sum += x[0] * y[0];
          x += strides_x[0];
          y += strides_y[0];
        }
      }
      d[0] = sum;
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
    using value_type = std::iter_value_t<std::ranges::iterator_t<O>>;

    auto first_d = std::ranges::begin(out);
    auto first_x = std::ranges::cbegin(in1);
    auto first_y = std::ranges::cbegin(in2);

    constexpr size_t type_size = sizeof(std::iter_value_t<std::ranges::iterator_t<O>>);

    constexpr size_t rank_d = std::ranges::iterator_t<O>::rank;
    constexpr size_t rank_x = std::ranges::iterator_t<I1>::rank;
    constexpr size_t rank_y = std::ranges::iterator_t<I2>::rank;
    // constexpr size_t rank = std::max(rank_x, rank_y);

    auto strides_d = first_d.Strides();
    auto strides_x = first_x.Strides();
    auto strides_y = first_y.Strides();

    auto& CPU = libai::device::CPU::GetDevice();
    auto& queue = CPU.GetQueue();

    if constexpr (rank_d > 0)
    {
      // use "tiling" by using the max size / max threads, aligned to cache line
      size_t cache_line = std::hardware_destructive_interference_size;
      std::array<size_t, rank_d> sizes;
      sizes.fill(1);
      sizes[rank_d - 1] = ((first_d.Extents()[rank_d - 1] + cache_line - 1) & -cache_line) / type_size;
      queue.Enqueue(first_d.Extents(),
                    sizes,
                    Eval<value_type, rank_d, rank_x, rank_y>,
                    first_x.Extents(), first_y.Extents(),
                    &*first_d, &*first_x, &*first_y,
                    std::move(strides_d),
                    std::move(strides_x),
                    std::move(strides_y));
      queue.Sync();
    }
    else
    {
      std::array<size_t, 0> pos{};

      Eval<value_type, rank_d, rank_x, rank_y>(
          pos, first_d.Extents(), first_d.Extents(),
          first_x.Extents(), first_y.Extents(),
          &*first_d, &*first_x, &*first_y,
          std::move(strides_d),
          std::move(strides_x),
          std::move(strides_y));
    }
  }
};

} // end of namespace libai

#endif  // GRID_TENSOR_CPU_MATMUL_H
