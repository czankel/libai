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

namespace libai {

/// UnaryOperation<Operator> implements element-wise unary operation on a tensors.
///
///  @tparm TOperator unary operator
template <template <typename> typename TOperator>
class UnaryOperation<TOperator, device::CPU>
{
  template <typename T, size_t N, bool Contiguous>
  static inline void Eval(std::span<const size_t, N> pos,
                          std::span<const size_t, N> dimensions,
                          std::span<const size_t, N> sizes,
                          T* d, const T* x,
                          std::span<const ssize_t, N> strides_d,
                          std::span<const ssize_t, N> strides_x)
  {
    if constexpr (dimensions.size() == 1 && Contiguous)
    {
      for (size_t i = pos[0] * sizes[0]; i < (pos[0] + 1) * sizes[0] && i < dimensions[0]; i++)
        d[i] = TOperator<device::CPU>()(x[i]);
    }
    else if constexpr (dimensions.size() == 1)
    {
      for (size_t i = pos[0] * sizes[0]; i < (pos[0] + 1) * sizes[0] && i < dimensions[0]; i++)
        d[i * strides_d[0]] = TOperator<device::CPU>()(x[i * strides_x[0]]);
    }
    else
    {
      size_t offset = pos[0] * sizes[0];
      d += offset * strides_d[0];
      x += offset * strides_x[0];

      for (size_t i = offset; i < offset + sizes[0] && i < dimensions[0]; i++)
      {
        Eval<T, N - 1, Contiguous>(pos.template last<N - 1>(),
                                   dimensions.template last<N - 1>(),
                                   sizes.template last<N - 1>(),
                                   d, x,
                                   strides_d.template last<N - 1>(),
                                   strides_x.template last<N - 1>());
        d += strides_d[0];
        x += strides_x[0];
      }
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

    Fold([&](const auto dimensions, const auto strides_d, const auto strides_x) {
        static_assert(dimensions.size() != std::dynamic_extent, "dynamic_extent not supported");

        using value_type = std::iter_value_t<std::ranges::iterator_t<O>>;
        constexpr size_t rank = dimensions.size();

        // special case: scaler
        if constexpr (rank == 0)
          *first_d = TOperator<device::CPU>()(*first_x);

        else
        {
          size_t type_size = sizeof(std::iter_value_t<std::ranges::iterator_t<O>>);
          bool is_cont = IsContiguous(strides_d, strides_x);

          const auto b_strides_x = BroadcastStrides<rank>(strides_x);

          auto& CPU = libai::device::CPU::GetDevice();
          auto& queue = CPU.GetQueue();

          // use "tiling" by using the max size / max threads, aligned to cache line
          size_t cache_line = std::hardware_destructive_interference_size; // FIXME: use device??
          std::array<size_t, rank> sizes;
          sizes.fill(1);
          sizes[rank - 1] = ((dimensions[rank - 1] + cache_line - 1) & -cache_line) / type_size;

          if (is_cont)
            queue.Enqueue(dimensions, sizes, Eval<value_type, rank, true>,
                          &*first_d, &*first_x, std::move(strides_d), std::move(b_strides_x));
          else
            queue.Enqueue(dimensions, sizes, Eval<value_type, rank, false>,
                          &*first_d, &*first_x, std::move(strides_d), std::move(b_strides_x));

          queue.Sync();
        }
    }, first_d.Extents(), first_d.Strides(), first_x.Strides());
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


} // end of namespace libai

#endif // GRID_TENSOR_CPU_UNARY_H
