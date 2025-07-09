//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_TENSOR_PARAMETERS_H
#define GRID_TENSOR_TENSOR_PARAMETERS_H

#include <algorithm>
#include <utility>

#include "concepts.h"

namespace grid {

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
template <size_t TRank, typename Indices = std::make_index_sequence<TRank>>
std::array<ssize_t, TRank> make_strides(const std::array<size_t, TRank>& dimensions)
{
  std::array<ssize_t, TRank> strides;
  ssize_t stride = 1;
  for (int i = static_cast<int>(TRank) - 1; i >= 0; i--)
  {
    strides[i] = dimensions[i] != 1 ? stride : 0;
    stride *= dimensions[i];
  }
  return strides;
}


// get_array_size returns the required size for the given dimensions and strides in numner of elements.
template <typename T, typename U, typename V>
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

/// @brief Helper function to align strides returning an array if a stride needs to be extended.
template <size_t S1, size_t S2>
inline auto BroadcastStrides(std::span<const ssize_t, S1> strides1, std::span<const ssize_t, S2> strides2)
{
  if constexpr (S1 == S2)
    return std::make_tuple(std::move(strides1), std::move(strides2));
  else if constexpr (S1 == 0)
    return std::make_tuple(std::move(std::array<const ssize_t, S2>{}), std::move(strides2));
  else if constexpr (S2 == 0)
    return std::make_tuple(std::move(strides1), std::move(std::array<const ssize_t, S1>{}));
  else if constexpr (S2 > S1)
  {
    std::array<ssize_t, S2> strides{};
    std::ranges::copy(strides1, strides.begin() + S2 - S1);
    return std::make_tuple(std::move(strides), std::move(strides2));
  }
  else
  {
    std::array<ssize_t, S1> strides{};
    std::ranges::copy(strides2, strides.begin() + S1 - S2);
    return std::make_tuple(std::move(strides1), std::move(strides));
  }
}

// TODO: the CUDA nvcc compiler doesn't support trailing return types
#if !defined(__CUDACC__)


/// @brief Helper function to check if all provided strides are contiguous (strides[last] == 1)
inline bool IsContiguous(auto... strides)
{
  return ((strides.size() > 0 && strides[strides.size() - 1] == 1) && ...);
}

/// @brief Helper function to reduce the rank for contiguous data
///
/// The Fold function calls the provided function with the folded dimensions and strides.
/// Strides are reduced by the folded dimensions. An empty span is used if the original
/// rank of the stride was smaller than the residual rank.
///
/// Folding rules:
///
/// Having a dimensions of 1 can always be folded, strides can be ignored
///   (1) dim:      _,1  -> fold
///
/// Dimensions can be folded if one of the following conditions is true:
///   (2) stride: [0],0  -> foldable, the sclar is applied to any folded dimensions)
///   (3) stride: f*x,f  -> foldable, the upper stride must match the "folded" dimension (x)
///                         multiplied by the lower stride)
///
template <size_t R, typename TOp>
void Fold(TOp&& op, std::span<const size_t, R> dims, auto... strides)
{
  // rank-0: scalars return empty dimensions, and are 'contiguous'
  if constexpr (R == 0)
    std::apply(op, std::tuple_cat(std::tuple(std::span<const size_t, 0>{}),
                                  std::array<std::span<const ssize_t, 0>, sizeof...(strides)>{}));

  // rank-1: scalar if dim is one, otherwise, keep strides
  else if constexpr (R == 1)
    if (dims[0] == 1)
      std::apply(op, std::tuple_cat(std::tuple(std::span<const size_t, 0>{}),
                                    std::array<std::span<const ssize_t, 0>, sizeof...(strides)>{}));
    else
      op(dims, strides...);

  else
  {
    size_t skip_idx = 0;  // track index for the first non-broadcast stride (i.e. dim != 1)
    size_t prev_idx = 0;  // track stride index to skip_idx "broadcast" entries (i.e. where dim == 1)
    size_t folded_dim = 1;

    auto foldfn = [&]<size_t I>() -> bool
    {
      // find first non-broadcast index (dim != 1)
      size_t dim = dims[R - I - 1];
      if (skip_idx == 0)
      {
        if (dim == 1)
          return true;
        skip_idx = I + 1;
        prev_idx = I + 1;
      }

      folded_dim *= dim;
      if constexpr (I < R - 1) // R - I > 1
      {
        bool foldable = dims[R - I - 2] == 1;
        if (!foldable)
        {
          foldable = (... && ([&](auto s) mutable {
              constexpr size_t sz = s.size();
              ssize_t first_str = strides[sz - skip_idx];
              ssize_t curr_str = strides[sz - I - 2];
              return ((sz >= I + 2 && curr_str - dims[R - prev_idx] * strides[sz - prev_idx] == 0) ||
                      ((sz < I + 2 || curr_str == 0) && (sz > skip_idx && first_str == 0)));
          }(strides)));
          prev_idx = I + 2;
        }

        if (foldable)
          return true;
      }

      std::array<size_t, R - I> folded_dims;
      std::ranges::copy(dims.template first<R - I - 1>(), folded_dims.begin());
      folded_dims[R - I - 1] = folded_dim;

      // TODO: check if folded_strides is dangling or lifetime extended
      // TODO: assumes lambda is called in reverse order of arguments!
      op(std::span(std::as_const(folded_dims)),
         std::span<const ssize_t, strides.size() <= I ? 0 : strides.size() - I>([&](auto s) {

          constexpr size_t sz = s.size();
          if constexpr (sz <= I)
            return std::span<const ssize_t, 0>{};
          else
          {
            std::array<ssize_t, sz - I> folded_strides{};
            std::ranges::copy(s.template first<sz - I>(), folded_strides.begin());
            if (sz > skip_idx)
              folded_strides[sz - I - 1] = s[sz - skip_idx];
            return std::as_const(folded_strides);
          }
      }(strides))...);
      return false;
    };

    [&] <std::size_t... I>(std::index_sequence<I...>)
    {
      if (((foldfn.template operator()<I>()) && ...))
        std::apply(op, std::tuple_cat(std::tuple(std::span<const size_t, 0>{}),
                                      std::array<std::span<const ssize_t, 0>, sizeof...(strides)>{}));
    }(std::make_index_sequence<R>{});
  }
}

/// @brief Helper function to reduce the rank for contiguous data and broadcasting strides.
///
/// The Fold function calls the provided function with the folded dimensions and strides.
/// Strides are "broadcast" to match the rank of the dimensions.
///
/// Any lower-rank provide stride will be "broadcasted" (setting additional dimensions to 0).
///
/// Note that right-hand-side vectors are "converted" to "row" vectors, while LHS vectors
/// are kept as "colunn" vectors. Thie means that row/col strides for RHS vectors get exchanged.
///
template <size_t R, typename TOp>
void FoldBroadcast(TOp&& op, std::span<const size_t, R> dims, auto... strides)
{
  // rank-0: scalars return empty dimensions, and are 'contiguous'
  if constexpr (R == 0)
    std::apply(op, std::tuple_cat(std::tuple(std::span<const size_t, 0>{}),
                                  std::array<std::span<const ssize_t, 0>, sizeof...(strides)>{}));

  // rank-1: scalar if dim is one, otherwise, keep vector and strides, expand stride to 0 if none
  else if constexpr (R == 1)
    if (dims[0] == 1)
      std::apply(op, std::tuple_cat(std::tuple(std::span<const size_t, 0>{}),
                                    std::array<std::span<const ssize_t, 0>, sizeof...(strides)>{}));
    else
    {
      const std::array<const ssize_t, 1> def_stride{0};
      op(dims, [&def_stride](auto s) -> std::span<const ssize_t, 1>{
          if constexpr (s.size() != 0)
            return s;
          else
            return def_stride;
        }(strides)...);
    }

  else
  {
    size_t skip_idx = 0;  // track index for the first non-broadcast stride (i.e. dim != 1)
    size_t prev_idx = 0;  // track stride index to skip_idx "broadcast" entries (i.e. where dim == 1)
    size_t folded_dim = 1;

    auto foldfn = [&]<size_t I>() -> bool
    {
      // find first non-broadcast index (dim != 1)
      size_t dim = dims[R - I - 1];
      if (skip_idx == 0)
      {
        if (dim == 1)
          return true;
        skip_idx = I + 1;
        prev_idx = I + 1;
      }

      folded_dim *= dim;
      if constexpr (I < R - 1) // R - I > 1
      {
        bool foldable = dims[R - I - 2] == 1;
        if (!foldable)
        {
          foldable = (... && ([&](auto s) mutable {
              constexpr size_t sz = s.size();
              ssize_t first_str = strides[sz - skip_idx];
              ssize_t curr_str = strides[sz - I - 2];
              return ((sz >= I + 2 && curr_str - dims[R - prev_idx] * strides[sz - prev_idx] == 0) ||
                      ((sz < I + 2 || curr_str == 0) && (sz > skip_idx && first_str == 0)));

          }(strides)));

          prev_idx = I + 2;
        }

        if (foldable)
          return true;
      }

      std::array<size_t, R - I> folded_dims;
      std::ranges::copy(dims.template first<R - I - 1>(), folded_dims.begin());
      folded_dims[R - I - 1] = folded_dim;

      // TODO: check if folded_strides is dangling or lifetime extended
      op(std::span(std::as_const(folded_dims)), std::span<const ssize_t, R - I>([&](auto s) {

          const size_t sz = s.size();

          std::array<ssize_t, R - I> folded_strides{};
          if constexpr (sz > I)
            std::ranges::copy(s.template first<sz - I>(), folded_strides.begin() + R - sz);
          if (sz > skip_idx)
            folded_strides[R - I - 1] = s[sz - skip_idx];

          return std::as_const(folded_strides);
      }(strides))...);

      return false;
    };

    [&] <std::size_t... I>(std::index_sequence<I...>)
    {
      if (((foldfn.template operator()<I>()) && ...))
        std::apply(op, std::tuple_cat(std::tuple(std::span<const size_t, 0>{}),
                                      std::array<std::span<const ssize_t, 0>, sizeof...(strides)>{}));
    }(std::make_index_sequence<R>{});
  }
}

#endif // !defined(__CUDACC__)

} // end of namespace grid

#endif  // GRID_TENSOR_TENSOR_PARAMETERS_H
