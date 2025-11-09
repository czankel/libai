//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef LIBAI_TENSOR_CPU_BINARY_H
#define LIBAI_TENSOR_CPU_BINARY_H

#include <span>
#include <algorithm>
#include <ranges>

#include "../binary.h"
#include "../concepts.h"
#include "../tensor_operation.h"

namespace libai {

/// BinaryOperation<Operator> implements element-wise binary operations of two tensors.
/// The dimensions of the tensors must match following broadcasting rules.
/// The resulting rank is the maximum of the tensor ranks.
///
///  @tparm TOperator binary operator
template <template <typename> typename TOperator>
class BinaryOperation<TOperator, device::CPU>
{
  template <typename T, size_t N, bool Contiguous>
  static inline void Eval(std::span<const size_t, N> pos,
                          std::span<const size_t, N> dimensions,
                          std::span<const size_t, N> sizes,
                          T* d, const T* x, const T* y,
                          std::span<const ssize_t, N> strides_d,
                          std::span<const ssize_t, N> strides_x,
                          std::span<const ssize_t, N> strides_y)
  {
    if constexpr (dimensions.size() == 1 && Contiguous)
    {
      for (size_t i = pos[0] * sizes[0]; i < (pos[0] + 1) * sizes[0] && i < dimensions[0]; i++)
        d[i] = TOperator<device::CPU>()(x[i], y[i]);
    }
    else if constexpr (dimensions.size() == 1)
    {
      for (size_t i = pos[0] * sizes[0]; i < (pos[0] + 1) * sizes[0] && i < dimensions[0]; i++)
        d[i * strides_d[0]] = TOperator<device::CPU>()(x[i * strides_x[0]], y[i * strides_y[0]]);
    }
    else
    {
      size_t offset = pos[0] * sizes[0];
      d += offset * strides_d[0];
      x += offset * strides_x[0];
      y += offset * strides_y[0];

      for (size_t i = offset; i < offset + sizes[0] && i < dimensions[0]; i++)
      {
        Eval<T, N - 1, Contiguous>(pos.template last<N - 1>(),
                                   dimensions.template last<N - 1>(),
                                   sizes.template last<N - 1>(),
                                   d, x, y,
                                   strides_d.template last<N - 1>(),
                                   strides_x.template last<N - 1>(),
                                   strides_y.template last<N - 1>());
        d += strides_d[0];
        x += strides_x[0];
        y += strides_y[0];
      }
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

    Fold([&](const auto dimensions, const auto strides_d, const auto strides_x, const auto strides_y) {
        static_assert(dimensions.size() != std::dynamic_extent, "dynamic_extent not supported");

        using value_type = std::iter_value_t<std::ranges::iterator_t<O>>;
        constexpr size_t rank = dimensions.size();

        // special case: scaler x scalar
        if constexpr (rank == 0)
          *first_d = TOperator<device::CPU>()(*first_x, *first_y);

        else
        {
          size_t type_size = sizeof(std::iter_value_t<std::ranges::iterator_t<O>>);
          bool is_cont = IsContiguous(strides_d, strides_x, strides_y);

          const auto [b_strides_x, b_strides_y] = BroadcastStrides<rank>(strides_x, strides_y);
          auto& CPU = libai::device::CPU::GetDevice();
          auto& queue = CPU.GetQueue();

          // use "tiling" by using the max size / max threads, aligned to cache line
          size_t cache_line = std::hardware_destructive_interference_size;
          std::array<size_t, rank> sizes;
          sizes.fill(1);
          sizes[rank - 1] = ((dimensions[rank - 1] + cache_line - 1) & -cache_line) / type_size;

          if (is_cont)
            queue.Enqueue(dimensions, sizes, Eval<value_type, rank, true>,
                          &*first_d, &*first_x, &*first_y,
                          std::move(strides_d),
                          std::move(b_strides_x),
                          std::move(b_strides_y));
          else
            queue.Enqueue(dimensions, sizes, Eval<value_type, rank, false>,
                          &*first_d, &*first_x, &*first_y,
                          std::move(strides_d),
                          std::move(b_strides_x),
                          std::move(b_strides_y));

          queue.Sync();
        }
    }, first_d.Extents(), first_d.Strides(), first_x.Strides(), first_y.Strides());
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


} // end of namespace libai

#endif // LIBAI_TENSOR_CPU_BINARY_H
