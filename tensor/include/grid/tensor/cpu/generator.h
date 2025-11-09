//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_CPU_GENERATOR_H
#define GRID_TENSOR_CPU_GENERATOR_H

#include <span>
#include <algorithm>
#include <ranges>

namespace libai {

template <> class GeneratorOperation<device::CPU>
{
  template <typename T, std::copy_constructible F>
  inline void generate(T* d,
                       std::span<const size_t, 1> dimensions,
                       std::span<const ssize_t, 1> strides,
                       F gen) const
  {
    // TODO: c++23 introduces an invoke_r: *d = invoke_r<T>(gen); should then work
    for (size_t i = 0; i < dimensions[0]; i++, d += strides[0])
      *d = gen.template operator()<T>();
  }

  template <typename T, size_t N, std::copy_constructible F>
  inline void generate(T* d,
                       std::span<const size_t, N> dimensions,
                       std::span<const ssize_t, N> strides,
                       F gen) const
  {
    for (size_t i = 0; i < dimensions[0]; i++, d += strides[0])
      generate(d,
               std::span<const size_t, N - 1>(dimensions.begin() + 1, dimensions.end()),
               std::span<const ssize_t, N - 1>(strides.begin() + 1, strides.end()),
               std::move(gen));
  }

 public:

  template<typename O, std::copy_constructible F>
  // TODO: requires invocable_r; what about result?
  // TODO std::invocable_r<R, F&> && std::ranges::output_range<O, std::invoke_result_t<F&>>
  void operator()(O&& out, F&& gen) const
  {
    using tensor_type = std::remove_cvref_t<O>;
    constexpr size_t rank = tensor_type::rank;
    auto fist_d = std::ranges::begin(out);

    generate(&*fist_d,
             std::span<const size_t, rank>{fist_d.Extents()},
             std::span{fist_d.Strides()},
             std::move(gen));
  }
};

//
// Operators
//

#if 0 // FIXME how to implement?
template <> struct FillOperator<device::CPU>
{
  template <typename T> inline void operator()() const { return val; }
};
#endif


template <> struct RandomFunction<device::CPU>
{
  template <typename T> inline T operator()() const
  {
    return static_cast<T>((static_cast<double>(std::rand()) / RAND_MAX) * T{100}); // FIXME need max
  }
};

} // end of namespace libai

#endif  // GRID_TENSOR_CPU_GENERATOR_H
