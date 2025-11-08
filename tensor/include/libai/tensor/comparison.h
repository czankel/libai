//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef LIBAI_TENSOR_COMPARISON_H
#define LIBAI_TENSOR_COMPARISON_H

#include "precision.h"

namespace libai {

template <typename T, size_t>
inline std::enable_if_t<!std::is_floating_point_v<T>, bool>
equals(const T* x, const T* y,
       std::span<const size_t,  0>,
       std::span<const ssize_t, 0>,
       std::span<const ssize_t, 0>)
{
  return *x == *y;
}

template <typename T, size_t>
inline std::enable_if_t<std::is_floating_point_v<T>, bool>
equals(const T* x, const T* y,
       std::span<const size_t,  0>,
       std::span<const ssize_t, 0>,
       std::span<const ssize_t, 0>)
{
  T eps = std::numeric_limits<T>::epsilon() * Precision::Margin();
  auto max = std::max(std::abs(*x), std::abs(*y));
  return std::abs(*x - *y) <= std::max(T{1}, max) * eps;
}

template <typename T, size_t>
inline std::enable_if_t<!std::is_floating_point_v<T>, bool>
equals(const T* x, const T* y,
       std::span<const size_t,  1> dimensions,
       std::span<const ssize_t, 1> strides_x,
       std::span<const ssize_t, 1> strides_y)
{
  for (size_t i = 0; i < dimensions[0]; i++)
  {
    if (*x != *y)
      return false;
    x += strides_x[0];
    y += strides_y[0];
  }
  return true;
}

template <typename T, size_t>
inline std::enable_if_t<std::is_floating_point_v<T>, bool>
equals(const T* x, const T* y,
       std::span<const size_t,  1> dimensions,
       std::span<const ssize_t, 1> strides_x,
       std::span<const ssize_t, 1> strides_y)
{
  for (size_t i = 0; i < dimensions[0]; i++)
  {
    T eps = std::numeric_limits<T>::epsilon() * Precision::Margin();
    auto max = std::max(std::abs(*x), std::abs(*y));

    if (std::abs(*x - *y) > std::max(T{1}, max) * eps)
      return false;

    x += strides_x[0];
    y += strides_y[0];
  }
  return true;
}

template <typename T, size_t N>
inline std::enable_if_t<(N > 1), bool>
equals(const T* x, const T* y,
       std::span<const size_t,  N> dimensions,
       std::span<const ssize_t, N> strides_x,
       std::span<const ssize_t, N> strides_y)
{
  static_assert(N != std::dynamic_extent, "dynamic_extent not allowed");
  for (size_t i = 0; i < dimensions[0]; i++)
  {
    if (!equals<T, N - 1>(x, y,
                             std::span<const size_t,  N - 1>(dimensions.begin() + 1, N - 1),
                             std::span<const ssize_t, N - 1>(strides_x.begin() + 1, N - 1),
                             std::span<const ssize_t, N - 1>(strides_y.begin() + 1, N - 1)))
      return false;

    x += strides_x[0];
    y += strides_y[0];
  }
  return true;
}

// TODO: will https://open-std.org/JTC1/SC22/WG21/docs/papers/2019/p1045r1.html help for using tensor.Rank() as constexpr?
template <PrimitiveTensor  TTensor1, PrimitiveTensor TTensor2>
bool operator==(TTensor1&& tensor1, TTensor2&& tensor2)
{
  constexpr size_t _Rank = std::remove_cvref_t<decltype(tensor1)>::rank;
  static_assert(_Rank == std::remove_cvref_t<decltype(tensor2)>::rank,
                "ranks mismatch between tensors");

  return equals<typename std::remove_cvref_t<TTensor1>::value_type, _Rank>(
                      tensor1.Data(),
                      tensor2.Data(),
                      std::span(tensor1.Dimensions()),
                      std::span(tensor1.Strides()),
                      std::span(tensor2.Strides()));
}

} // end of namespace libai

#endif // LIBAI_TENSOR_COMPARISON_H
