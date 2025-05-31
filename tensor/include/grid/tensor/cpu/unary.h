//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_CPU_UNARY_H
#define GRID_TENSOR_CPU_UNARY_H

#include <span>
#include <algorithm>
#include <ranges>

#include "../concepts.h"
#include "../unary.h"
#include "../tensor_operation.h"

namespace grid {

/// UnaryOperation<Operator> implements element-wise unary operation on a tensors.
///
///  @tparm TOperator unary operator
template <template <typename> typename TOperator>
class UnaryOperation<TOperator, device::CPU>
{

  // scalar operation
  template <typename T>
  inline void Eval(T* d, const T* x) const
  {
    d[0] = TOperator<device::CPU>()(x[0]);
  }

  // contiguous vector
  template <typename T>
  inline void Eval(T* d, const T* x, std::span<const size_t, 1> dimensions) const
  {
    for (size_t i = 0; i < dimensions[0]; i++) {
      d[i] = TOperator<device::CPU>()(x[i]);
    }
  }

  // discontiguous vector or scalar
  template <typename T>
  inline void Eval(T* d, const T* x,
                   std::span<const size_t, 1>  dimensions,
                   std::span<const ssize_t, 1> strides_d,
                   std::span<const ssize_t, 1> strides_x) const
  {
    for (size_t i = 0; i < dimensions[0]; i++)
      d[i * strides_d[0]] = TOperator<device::CPU>()(x[i * strides_x[0]]);
  }

  // rank 2 and greater, discontiguous
  template <typename T>
  inline void Eval(T* d, const T* x, auto dimensions, auto strides_d, auto strides_x) const
  {
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      Eval(d, x,
           dimensions.template last<dimensions.size() - 1>(),
           strides_d.template last<strides_d.size() - 1>(),
           strides_x.template last<strides_x.size() - 1>());
      d += strides_d[0];
      x += strides_x[0];
    }
  }

  // rank 2 and greater, contiguous
  template <typename T>
  inline void
  EvalContiguous(T* d, const T* x, auto dimensions, auto strides_d, auto strides_x) const
  {
    for (size_t i = 0; i < dimensions[0]; i++)
    {
      if constexpr (dimensions.size() > 2)
      {
        EvalContiguous(d, x,
             dimensions.template last<dimensions.size() - 1>(),
             strides_d.template last<strides_d.size() - 1>(),
             strides_x.template last<strides_x.size() - 1>());
        d += strides_d[0];
        x += strides_x[0];
      }
      else if constexpr (dimensions.size() > 1)
      {
        Eval(d, x,
             dimensions.template last<dimensions.size() - 1>(),
             strides_d.template last<strides_d.size() - 1>(),
             strides_x.template last<strides_x.size() - 1>());
        d += strides_d[0];
        x += strides_x[0];
      }
      else
        Eval(d, x, dimensions.template last<dimensions.size() - 1>());
    }
  }

 public:
  template<std::ranges::input_range I,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<I>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<I>, std::ranges::iterator_t<O>>
  void operator()(I&& in, O&& out) const
  {
    auto first_d = std::ranges::begin(out);
    auto first_x = std::ranges::cbegin(in);

    FoldBroadcast([&](auto dimensions, auto strides_d, auto strides_x) {
        static_assert(dimensions.size() != std::dynamic_extent, "dynamic_extent not supported");
        bool is_cont = IsContiguous(strides_d, strides_x);

        if constexpr (dimensions.size() == 0)
          Eval(&*first_d, &*first_x);
        else if constexpr (dimensions.size() == 1)
          if (is_cont)
            Eval(&*first_d, &*first_x, dimensions);
          else
            Eval(&*first_d, &*first_x, dimensions, strides_d, strides_x);
        else if (is_cont)
          EvalContiguous(&*first_d, &*first_x, dimensions, strides_d, strides_x);
        else
          Eval(&*first_d, &*first_x, dimensions, strides_d, strides_x);
    }, std::span(first_d.Extents()), std::span(first_d.Strides()), std::span(first_x.Strides()));
  }
};

//
// Elementary Unary Operators
//

template <> struct CopyOperator<device::CPU>
{
  template<typename T> inline T operator()(const T x) const { return x; }
};

template <> struct NegOperator<device::CPU>
{
  template<typename T> inline T operator()(const T x) const { return -x; }
};

//
// Unary Operations
//

template <> struct SiluFunction<device::CPU>
{
  template<typename T> inline T operator()(const T x) const { return x / (T{1} + exp(-x)); }
};


} // end of namespace grid

#endif // GRID_TENSOR_CPU_UNARY_H
