//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef TENSOR_SOURCE_CUDA_KERNELS_UTILS_H
#define TENSOR_SOURCE_CUDA_KERNELS_UTILS_H

#include <initializer_list>

namespace libai {
namespace cuda {

namespace
{
template <std::size_t ... Is>
constexpr auto index_sequence_reverse(std::index_sequence<Is...> const &)
  -> decltype(std::index_sequence<sizeof...(Is) - 1U - Is...>{});

template <std::size_t N>
using make_index_sequence_reverse = decltype(index_sequence_reverse(std::make_index_sequence<N>{}));
}

// The number of threads implemented in GPUs is currently fixed to 1024
static constexpr size_t MaxThreadCount = 1024;
static constexpr size_t WarpSize = 32;

// MakeDim3 is a helper function to create a dim3 value out of a span of n elements (n <=3)
template <typename T, size_t R>
inline dim3 MakeDim3(std::span<T, R> s)
{
  return [=] <std::size_t... I> (std::index_sequence<I...>) {
     return dim3{(static_cast<unsigned int>(s[I]))...};
  }(make_index_sequence_reverse<R>());
}

// GetSizes returns a tuple of the grid and block sizes for the provided dimension.
// Grid size is the number of thread blocks in the grid
// Block size is the number of threads in each thread block
inline auto GetSizes(size_t size)
{
  int block_size = std::min(size, MaxThreadCount);
  int grid_size = (int)ceil((float)size/block_size);
  return std::make_tuple(grid_size, block_size);
}

// GetSizes returns a tuple of the 3-dimensional grid and block sizes for the provided dimensions.
// The sizes are .. capped by the maximum thread count divided by..
// They should be chosen such that:
//  - the loads of all parallel threads should be limited to the same 1024k (one memory lane)
//    for a simple float addition d[i] = x[2*i] + x[2*i+1], this would be 2 (loads) x 4 (32-bit float)
//    the divisor should be 4 for the first
//
// Grid size is the number of thread blocks in grid
// Block size is the number of threads in each thread block
template <size_t R, typename... S>
inline auto GetSizes(std::span<const size_t, R> dims, S... sizes)
{
  return [=] <std::size_t... I> (std::index_sequence<I...>, auto sz) {
     return std::make_tuple(
         dim3{((unsigned int)ceil((float)(dims[I])/std::min(static_cast<size_t>(std::get<I>(sz)), MaxThreadCount)))...},
         dim3{((unsigned int)std::min(static_cast<size_t>(std::get<I>(sz)), MaxThreadCount))...});
  }(std::make_index_sequence<R>(), std::make_tuple(sizes...));
}

template <size_t R, typename T, typename... S>
inline auto GetSizes(const T(&&dims)[R], S... sizes)
{
  return [=] <std::size_t... I> (std::index_sequence<I...>, auto sz) {
     return std::make_tuple(
         dim3{((unsigned int)ceil((float)(dims[I])/std::min(static_cast<size_t>(std::get<I>(sz)), MaxThreadCount/sizeof(T))))...},
         dim3{((unsigned int)std::min(static_cast<size_t>(std::get<I>(sz)), MaxThreadCount/sizeof(T)))...});
  }(std::make_index_sequence<R>(), std::make_tuple(sizes...));
}



} // end of namespace cuda
} // end of namespace libai

#endif  // TENSOR_SOURCE_CUDA_KERNELS_UTILS_H
