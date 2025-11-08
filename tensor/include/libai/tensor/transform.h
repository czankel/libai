//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef LIBAI_TENSOR_TRANSFORM_H
#define LIBAI_TENSOR_TRANSFORM_H

#include <span>
#include <algorithm>
#include <ranges>

#include "concepts.h"

namespace libai {


/// @brief TransformFunction provides helper for applying an operator on a range(s)
///
/// TransfrormFunction is a class that provides overloaded operators operator() for
/// applying an operator on a range (Tensor) or ranges (Tensors) and stores the result
/// in another provide range (Tensor). The class shouldn't be used directly but through
/// the Transorm function.
///
/// @see Transform()
///
/// Example:
///
///   Transform(tensor1, tensor2, result.begin(), BinaryOperation<AddOperator>{});
class TransformFunction
{
 public:

  // in-out
  template<std::input_iterator I, std::sentinel_for<I> S,
           std::weakly_incrementable O,
           std::copy_constructible F>
  requires std::indirectly_copyable<I, O>
  constexpr std::ranges::unary_transform_result<I, O>
  operator()(I first, S last, O result, F op) const
  {
    constexpr size_t rank = O::rank;

    // TODO, identify if first is {0} and skip loop
    auto dimensions = last.Coordinates();
    auto& subtrahend = first.Coordinates();
    for (size_t i = 0; i < rank; i++)
      dimensions[i] -= subtrahend[i];

    std::invoke(op, &*result, &*first, dimensions, result.Strides(), first.Strides());
    first += dimensions;
    result += dimensions;

    return {std::move(first), std::move(result)};
  }

  // in-out
  template<std::ranges::input_range R,
           std::weakly_incrementable O,
           std::copy_constructible F>
  requires std::indirectly_copyable<std::ranges::iterator_t<R>, O>
  constexpr std::ranges::unary_transform_result<std::ranges::borrowed_iterator_t<R>, O>
  operator()(R&& r, O result, F op) const
  {
    return (*this)(std::ranges::begin(r), std::ranges::end(r),
                   std::move(result),
                   std::ref(op));
  }

  // in-in-out
  template<std::input_iterator I1, std::sentinel_for<I1> S1,
           std::input_iterator I2, std::sentinel_for<I2> S2,
           std::weakly_incrementable O,
           std::copy_constructible F>
  requires std::indirectly_copyable<I1, O> && std::indirectly_copyable<I2, O>
  // FIXME && IsDevice<I, BaseCPU> ??
  constexpr std::ranges::binary_transform_result<I1, I2, O>
  operator()(I1 first1, S1 last1, I2 first2, S2 last2, O result, F op) const
  {
    constexpr size_t rank = O::rank;

    // TODO, identify if first is {0} and skip loop
    auto dimensions = last1.Coordinates();
    auto& subtrahend = first1.Coordinates();
    for (size_t i = 0; i < rank; i++)
      dimensions[i] -= subtrahend[i];

    std::invoke(op, &*result, &*first1, &*first2, dimensions, result.Strides(), first1.Strides(), first2.Strides());
    first1 += dimensions;
    first2 += dimensions;
    result += dimensions;

    return {std::move(first1), std::move(first2), std::move(result)};
  }

  // in-in-out
  template<std::ranges::input_range R1,
           std::ranges::input_range R2,
           std::weakly_incrementable O,
           std::copy_constructible F>
  requires std::indirectly_copyable<std::ranges::iterator_t<R1>, O> &&
           std::indirectly_copyable<std::ranges::iterator_t<R2>, O>
  constexpr std::ranges::binary_transform_result<std::ranges::borrowed_iterator_t<R1>,
                                                 std::ranges::borrowed_iterator_t<R2>, O>
  operator()(R1&& r1, R2&& r2, O result, F op) const
  {
    return (*this)(std::ranges::begin(r1), std::ranges::end(r1),
                   std::ranges::begin(r2), std::ranges::end(r2),
                   std::move(result),
                   std::ref(op));
  }
};


/// Transform applies the given operator to a provided range (Tensor) or ranges (Tensors)
/// and stores the result in another provided range (Tensor).
inline constexpr TransformFunction Transform;


} // end of namespace libai

#endif  // LIBAI_TENSOR_TRANSFORM_H
