//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef LIBAI_TENSOR_TENSOR_PARAMETERS_H
#define LIBAI_TENSOR_TENSOR_PARAMETERS_H

#include <array>
#include <algorithm>
#include <stdexcept>
#include <utility>

#include "concepts.h"

namespace libai {

// helper class to determine rank from a reference; constexpr cannot bind to a reference
namespace
{
template <typename> struct container_template_args;
template <template <typename, size_t> typename C, typename T, size_t R>
struct container_template_args<C<T, R>> { using value_type = T; constexpr static size_t size = R; };
}


// get_array(iniitlizer_list)
// returns a std::array initialized from a initializer list.
template <typename T, size_t... Ns>
inline constexpr std::array<T, sizeof...(Ns)>
get_array_impl(std::initializer_list<T>&& init, std::index_sequence<Ns...>)
{
  return std::array<T, sizeof...(Ns)>{ *(init.begin() + Ns) ... };
}

template <typename T, size_t N, typename Ns = std::make_index_sequence<N>>
inline constexpr std::array<T, N>
get_array(std::initializer_list<T>&& init)
{
  return get_array_impl(std::move(init), Ns{});
}

// get_array(initializer_list<initializer_list>)
// returns a std::array from a 2-dimensional initializer list.
template <typename T, size_t M, size_t N>
inline constexpr std::array<T, M * N>
get_array(std::initializer_list<std::initializer_list<T>>&& init)
{
  std::array<T, M * N> arr{};
  auto line_it = arr.begin();
  for (auto it : init)
  {
    std::copy(it.begin(), it.end(), line_it);
    line_it += N;
  }
  return arr;
}

// get_array(initializer_list<initializer_list<initializer_list>>)
// returns a std::array from a 3-dimensional initializer list.
template <typename T, size_t C, size_t M, size_t N>
inline constexpr std::array<T, C * M * N>
get_array(std::initializer_list<std::initializer_list<std::initializer_list<T>>>&& init)
{
  std::array<T, C * M * N> arr{};
  auto line_it = arr.begin();
  for (auto lt : init)
  {
    for (auto it : lt)
    {
      std::copy(it.begin(), it.end(), line_it);
      line_it += N;
    }
  }
  return arr;
}

// get_array(T(&)[])
// returns a std::array from a c-array.
template <Arithmetic T, size_t N>
inline constexpr std::array<T, N>
get_array(const T(&init)[N])
{
  std::array<T, N> arr{};
  std::copy(std::begin(init), std::end(init), arr.begin());
  return arr;
}

// get_array(T(&&)[])
// returns a std::array from a c-array (rvalue reference)
template <Arithmetic T, size_t N>
inline constexpr std::array<T, N>
get_array(T(&&init)[N])
{
  std::array<T, N> arr{};
  std::copy(std::begin(init), std::end(init), arr.begin());
  return arr;
}

// get_array(T(&&...)[N])
// returns a std::array from a 2-dimensional c-array
template <Arithmetic T, size_t... N>
inline constexpr std::array<T, sizeof...(N) * std::max({N...})>
get_array(T(&&... init)[N])
{
  constexpr size_t maxcols = std::max({N...});
  std::array<T, sizeof...(N) * maxcols> arr{};
  auto line_it = arr.begin();

  auto apply = [&] <typename U> (U&& value, size_t cols) -> void {
    for (size_t i = 0; i < cols; i++, ++line_it)
      *line_it = value[i];
    line_it += maxcols - cols;
  };

  (apply(std::forward<T[N]>(init), N),...);
  return arr;
}

// get_array(T((&&...)[M])[N])
// returns a std::array from a 3-dimensional c-array
template <Arithmetic T, size_t... M, size_t... N>
inline constexpr std::array<T, sizeof...(M) * std::max({M...}) * std::max({N...})>
get_array(T((&&... init)[M])[N])
{
  constexpr size_t maxrows = std::max({M...});
  constexpr size_t maxcols = std::max({N...});
  std::array<T, sizeof...(M) * maxrows * maxcols> arr{};
  auto line_it = arr.begin();

  auto apply = [&] <typename U> (U&& value, size_t rows, size_t cols) -> void {
    for (size_t i = 0; i < rows; i++, line_it += maxcols - cols)
      for (size_t j = 0; j < cols; j++, ++line_it)
        *line_it = value[i][j];
    line_it += maxrows - rows;
  };

  (apply(std::forward<T[M][N]>(init), M, N),...);
  return arr;
}


// make_strides returns a std::array with the strides calculated from the provided dimensions and
// the template type parameter (make_strides<TYPE>(...))
template <size_t NRank, typename Indices = std::make_index_sequence<NRank>>
std::array<ssize_t, NRank> make_strides(const std::array<size_t, NRank>& dimensions)
{
  std::array<ssize_t, NRank> strides;
  ssize_t stride = 1;
  for (int i = static_cast<int>(NRank) - 1; i >= 0; i--)
  {
    strides[i] = dimensions[i] != 1 ? stride : 0;
    stride *= dimensions[i];
  }
  return strides;
}


// get_array_size returns the required size for the given dimensions and strides in numner of elements.
template <typename U, typename V>
size_t get_array_size(U&& dimensions, V&& strides)
{
  size_t size = 1;  // default is rank-0, which has size 1
  auto di = std::forward<U>(dimensions).begin();
  auto si = std::forward<V>(strides).begin();
  for (; di != dimensions.end() && si != strides.end(); ++di, ++si)
    size = std::max(size, *di * *si);
  return size;
}


// get_buffer_size returns the required size for the given dimensions and strides.
template <typename T, typename U, typename V>
size_t get_buffer_size(U&& dimensions, V&& strides)
{
  size_t size = 1;  // default is rank-0, which has size 1
  auto di = std::forward<U>(dimensions).begin();
  auto si = std::forward<V>(strides).begin();
  for (; di != dimensions.end() && si != strides.end(); ++di, ++si)
    size = std::max(size, *di * *si);
  return size * sizeof(T);
}


// @brief Calculate the size of an (sub-)area with the given dimensions within a larger tensor.
template <typename T, typename U, typename V>
size_t get_block_size(U&& dimensions, V&& strides)
{
  size_t size = 1;  // default is rank-0, which has size 1
  auto di = std::forward<U>(dimensions).begin();
  auto si = std::forward<V>(strides).begin();
  for (; di != dimensions.end() && si != strides.end(); ++di, ++si)
    size += std::max(*di - 1, 0UL) * *si;
  return size;
}


// Broadcast expands dimensions ("broadcasting") of the left tensor to match the right tensor
template <typename TTensor1, typename TTensor2>
inline auto BroadcastDimensions(const TTensor1& tensor1, const TTensor2& tensor2)
{
  constexpr size_t rank1 = TTensor1::rank;
  constexpr size_t rank2 = TTensor2::rank;
  constexpr size_t drank1 = rank2 > rank1 ? rank2 - rank1 : 0UL;
  constexpr size_t drank2 = rank1 > rank2 ? rank1 - rank2 : 0UL;

  if constexpr (rank1 == 0)
    return tensor2.Dimensions();
  else if constexpr (rank2 == 0)
    return tensor1.Dimensions();
  else
  {
    const auto& dimensions1 = tensor1.Dimensions();
    const auto& dimensions2 = tensor2.Dimensions();

    std::array<size_t, std::max(rank1, rank2)> dimensions;
    std::generate(dimensions.begin(), dimensions.end(), [n = 0, &dimensions1, &dimensions2]() mutable -> size_t
    {
      size_t k = n++;
      if (k < drank1 || dimensions1[k - drank1] == 1)
        return dimensions2[k - drank2];
      else if (k < drank2 || dimensions2[k - drank2] == 1 || dimensions2[k - drank2] == dimensions1[k - drank1])
        return dimensions1[k - drank1];
      else
        throw std::runtime_error("broadcast failed");
    });
    return dimensions;
  }
}

/// @brief Broadcast (expand the dimensions of the strides) to the provided rank R by inserting zeros.
//
/// Returns the provided strides directly if the rank is already R or a new std::array with the
/// original strides copied to the right-most dimensions.
///
/// The array size must be at most R or an assertion error is generated.
///
/// @param<in> strides... list of strides with
template <size_t R, typename... Ts>
requires (sizeof...(Ts) > 1)
inline auto BroadcastStrides(Ts&&... strides)
{
  return std::make_tuple([&]() {
    using type = std::remove_cvref_t<Ts>;
    using value_type = type::value_type;

    constexpr size_t rank = container_template_args<std::decay_t<Ts>>::size;
    static_assert(rank <= R);

    if constexpr (rank == R)
      return std::forward<Ts>(strides);
    else if constexpr (rank < R)
    {
      std::array<value_type, R> res{};
      std::ranges::copy(strides, res.begin() + R - strides.size());
      return res;
    }
  }() ...);
}

template <size_t R, typename T>
inline auto BroadcastStrides(T&& strides)
{
    using type = std::remove_cvref_t<T>;
    using value_type = type::value_type;

    // compiler complains about strides.size() not being constexpr
    constexpr size_t rank = std::remove_cvref_t<T>{}.size();
    static_assert(rank <= R);

    if constexpr (rank == R)
      return std::forward<T>(strides);
    else if constexpr (rank < R)
    {
      std::array<value_type, R> res{};
      std::ranges::copy(strides, res.begin() + R - strides.size());
      return res;
    }
}

// TODO: the CUDA nvcc compiler doesn't support trailing return types
#if !defined(__CUDACC__)

/// @brief Helper function to check if all provided strides are contiguous (strides[last] == 1)
template <typename... Ts>
inline bool IsContiguous(Ts&&... strides)
{
  return ((std::forward<Ts>(strides).size() > 0 && std::forward<Ts>(strides)[std::forward<Ts>(strides).size() - 1] == 1) && ...);
}

/// @brief Helper function to reduce the rank for contiguous data
///
/// The Fold function calls the provided function with the folded dimensions and strides.
/// Strides are reduced by the folded dimensions. The stride array is extended with 0s
/// if the original rank of the stride was smaller than the residual rank unless it was empty,
/// in which case it returned an empty array.
///
/// Folding rules:
///
/// Having a dimensions of 1 can always be folded, strides can be ignored
///   (1) dim:      _,1  -> fold
///
/// Dimensions can be folded if one of the following conditions is true:
///   (2) stride: f*x,f  -> foldable, the upper stride must match the "folded" dimension (x)
///                         multiplied by the lower stride)
///   (3) stride: [0],0  -> note: special case of (2)
///

template <typename TOp, typename T>
void Fold(TOp&& op, T&& dims, const auto... strides)
{
  // TODO: constexpr size_t rank = dims.size()
  constexpr size_t rank = container_template_args<std::decay_t<T>>::size;

  // rank-0: scalars return empty dimensions and strides
  if constexpr (rank == 0)
    std::apply(op, std::tuple_cat(std::tuple(std::array<size_t, 0>{}),
                                  std::array<std::array<ssize_t, 0>, sizeof...(strides)>{}));

  // rank-1: scalar if dim is one, otherwise, keep strides
  else if constexpr (rank == 1)
    if (dims[0] == 1)
      std::apply(op, std::tuple_cat(std::tuple(std::array<size_t, 0>{}),
                                    std::array<std::array<ssize_t, 0>, sizeof...(strides)>{}));
    else
      op(dims, std::move(strides)...);

  // rank-2 and higher
  else
  {
    size_t folded_dim = 1;
    size_t last = 0;  // track index from right to left, skip indices where the dimension is 1
    size_t skip = 0;  // first rank from the right where dimensions is not 1

    auto foldfn = [&]<size_t I>() -> bool
    {
      size_t dim = dims[rank - I - 1];
      if (skip == 0)
      {
        if (dim == 1)
          return true;
        skip = I + 1;
        last = rank - I - 1;
      }

      folded_dim *= dim;

      if constexpr (I < rank - 1)
      {
        bool foldable = dims[rank - I - 2] == 1;
        if (!foldable)
        {
          foldable = (... && ([&]() {
              constexpr int sz = strides.size();
              return (sz < I + 2) ?
                (sz < skip || strides[sz - skip] == 0) :
                ((strides[sz - I - 2] - dims[last] * strides[last + sz - rank]) == 0);
          }()));
          last = rank - I - 2;
        }

        if (foldable)
          return true;
      }

      std::array<size_t, rank - I> folded_dims;
      std::copy(dims.begin(), dims.begin() + rank - I, folded_dims.begin());
      folded_dims[rank - I - 1] = folded_dim;

      op(std::move(folded_dims), [&](auto s) {
            constexpr size_t sz = s.size();
            if constexpr (sz <= I)
              return std::array<ssize_t, 0>{};
            else
            {
              std::array<ssize_t, rank - I> folded_strides{};
              std::copy(s.begin(), s.begin() + sz - I, folded_strides.begin() + rank - sz);
              if (sz > skip)
                folded_strides[rank - I - 1] = s[sz - skip];
              return folded_strides;
            }
          }(strides)...);
      return false;
    };

    [&]<std::size_t... I>(std::index_sequence<I...>) {
      if (((foldfn.template operator()<I>()) && ...))
        std::apply(op, std::tuple_cat(std::tuple(std::array<size_t, 0>{}),
                                      std::array<std::array<ssize_t, 0>, sizeof...(strides)>{}));
    }(std::make_index_sequence<rank>{});
  }
}

#endif // !defined(__CUDACC__)

} // end of namespace libai

#endif  // LIBAI_TENSOR_TENSOR_PARAMETERS_H
