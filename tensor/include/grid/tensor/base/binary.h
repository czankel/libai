//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_CPU_BINARY_H
#define GRID_TENSOR_CPU_BINARY_H

#include <span>
#include <algorithm>
#include <ranges>

#include "../binary.h"
#include "../concepts.h"
#include "../tensor_operation.h"

namespace grid {

/// BinaryOperation<Operator> implements element-wise binary operations of two tensors.
/// The dimensions of the tensors must match following broadcasting rules.
/// The resulting rank is the maximum of the tensor ranks.
///
///  @tparm TOperator binary operator
template <template <typename> typename TOperator>
class BinaryOperation<TOperator, device::CPU>
{
  // TODO: gcc doesn't like this constexpr, which would be use later as just Operator(args).
  // Should it? See P0386R2 change: 9.2.3.2p3
  // static constexpr TOperator<device::CPU> Operator;

  // scalar operation
  template <typename T>
  inline void Eval(T* d, const T* x, const T* y) const
  {
    d[0] = TOperator<device::CPU>()(x[0], y[0]);
  }

  // contiguous vector
  template <typename T>
  inline void Eval(T* d, const T* x, const T* y, std::span<const size_t, 1> dimensions) const
  {
    for (size_t i = 0; i < dimensions[0]; i++)
      d[i] = TOperator<device::CPU>()(x[i], y[i]);
  }

  // discontiguous vector or scalar
  template <typename T>
  inline void Eval(T* d, const T* x, const T* y,
                   std::span<const size_t, 1>  dimensions,
                   std::span<const ssize_t, 1> strides_d,
                   std::span<const ssize_t, 1> strides_x,
                   std::span<const ssize_t, 1> strides_y) const
  {
    for (size_t i = 0; i < dimensions[0]; i++)
      d[i * strides_d[0]] = TOperator<device::CPU>()(x[i * strides_x[0]], y[i * strides_y[0]]);
  }

  // rank 2 and greater, discontiguous
  template <typename T>
  inline void Eval(T* d, const T* x, const T* y,
                   auto dimensions, auto strides_d, auto strides_x, auto strides_y) const
  {
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      Eval(d, x, y,
           dimensions.template last<dimensions.size() - 1>(),
           strides_d.template last<strides_d.size() - 1>(),
           strides_x.template last<strides_x.size() - 1>(),
           strides_y.template last<strides_y.size() - 1>());
      d += strides_d[0];
      x += strides_x[0];
      y += strides_y[0];
    }
  }

  // rank 2 and greater, contiguous
  template <typename T>
  inline void EvalContiguous(T* d, const T* x, const T* y,
                             auto dimensions, auto strides_d, auto strides_x, auto strides_y) const
  {
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      if constexpr (dimensions.size() > 2)
      {
        EvalContiguous(d, x, y,
             dimensions.template last<dimensions.size() - 1>(),
             strides_d.template last<strides_d.size() - 1>(),
             strides_x.template last<strides_x.size() - 1>(),
             strides_y.template last<strides_y.size() - 1>());
        d += strides_d[0];
        x += strides_x[0];
        y += strides_y[0];
      }
      else if constexpr (dimensions.size() > 1)
      {
        Eval(d, x, y, dimensions.template last<dimensions.size() - 1>());
        d += strides_d[0];
        x += strides_x[0];
        y += strides_y[0];
      }
      else
        Eval(d, x, y, dimensions.template last<dimensions.size() - 1>());
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

    FoldBroadcast([&](auto dimensions, auto strides_d, auto strides_x, auto strides_y) {
        static_assert(dimensions.size() != std::dynamic_extent, "dynamic_extent not supported");
        bool is_cont = IsContiguous(strides_d, strides_x, strides_y);

        if constexpr (dimensions.size() == 0)
          Eval(&*first_d, &*first_x, &*first_y);
        else if constexpr (dimensions.size() == 1)
          if (is_cont)
            Eval(&*first_d, &*first_x, &*first_y, dimensions);
          else
            Eval(&*first_d, &*first_x, &*first_y, dimensions, strides_d, strides_x, strides_y);
        else if constexpr (dimensions.size() > 1)
        {
          if (is_cont)
            EvalContiguous(&*first_d, &*first_x, &*first_y, dimensions, strides_d, strides_x, strides_y);
          else
            Eval(&*first_d, &*first_x, &*first_y, dimensions, strides_d, strides_x, strides_y);
        }
    }, std::span(first_d.Extents()), std::span(first_d.Strides()),
       std::span(first_x.Strides()), std::span(first_y.Strides()));
  }
};

//
// Elementary Binary Operators
//

template<> struct AddOperator<device::CPU>
{
  template<typename T> inline T operator()(T a, T b) const { return a + b; }
};

template<> struct SubOperator<device::CPU>
{
  template<typename T> inline T operator()(T a, T b) const { return a - b; }
};

template<> struct MulOperator<device::CPU>
{
  template<typename T> inline T operator()(T a, T b) const { return a * b; }
};

template<> struct DivOperator<device::CPU>
{
  template<typename T> inline T operator()(T a, T b) const { return a / b; }
};


} // end of namespace grid

#endif // GRID_TENSOR_CPU_BINARY_H
