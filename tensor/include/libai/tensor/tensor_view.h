//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef LIBAI_TENSOR_TENSOR_VIEW_H
#define LIBAI_TENSOR_TENSOR_VIEW_H

#include <algorithm>

#include "array.h"
#include "iterator.h"

namespace libai {

namespace {

// pointer_cast provides a (value) const-qualified pointer cast
template <typename T, typename S> requires (std::is_const_v<std::remove_pointer_t<S>>)
const std::remove_pointer_t<T>* pointer_cast(S pointer)
{
  return reinterpret_cast<const std::remove_pointer_t<T>*>(pointer);
}

template <typename T, typename S> requires (!std::is_const_v<std::remove_pointer_t<S>>)
std::remove_pointer_t<T>* pointer_cast(S pointer)
{
  return reinterpret_cast<std::remove_pointer_t<T>*>(pointer);
}

} // end of namespace


/// TensorView<Tensor, Rank> implements a view of a tensor.
///
/// Note that a view cannot be created from a temporary rval; it will return a tensor.
template <PrimitiveTensor TTensor, size_t TViewRank>
class TensorView
{
 public:
  using value_type = typename TTensor::value_type;
  using memory_type = typename TTensor::memory_type;
  using pointer = decltype(std::declval<TTensor>().Data());
  using reference = decltype(*std::declval<TTensor>().Data());
  using const_pointer = typename TTensor::const_pointer;
  using const_reference = typename TTensor::const_reference;
  using array_type = typename TTensor::array_type;
  constexpr static size_t rank = TViewRank;

  /// Default constructor
  TensorView() = default;

  /// Constructor with arguments
  TensorView(TTensor& tensor,
             const std::array<size_t,  TViewRank>& dimensions,
             const std::array<ssize_t, TViewRank>& strides,
             size_t size,
             size_t offset = 0UL)
    : tensor_(tensor),
      dimensions_(dimensions),
      strides_(strides),
      size_(size),
      offset_(offset)
  {
    if (offset_ + size_ > tensor.Size())
      throw std::runtime_error("view parameters exceed the tensor size");
  }


  /// operator=(Tensor) copies data from the rhs tensor (or view) into the view of the dependent tensor.
  template <AnyTensor TFromTensor> requires (TFromTensor::rank == TViewRank)
  auto operator=(const TFromTensor& rhs)
  {
    Copy(*this, rhs);
    return *this;
  }

  template <AnyOperator TOperator> // FIXME requires PrimitiveTensor<to_tensor(TOperator)>
  auto operator=(const TOperator& oper)
  {
    return operator=(oper());
  }


  /// begin returns an iterator for the begin of the Tensor array
  auto begin()                        { return details::Iterator(*this); }

  /// end returns the sentinel for the end of the Tensor array
  auto end()                          { return details::Iterator(*this, Dimensions()); }

  /// begin returns an iterator for the begin of the Tensor array
  auto begin() const                  { return details::ConstIterator(*this); }

  /// end returns the sentinel for the end of the Tensor array
  auto end() const                    { return details::ConstIterator(*this, Dimensions()); }


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return TViewRank; }

  /// Dimensions returns the dimensions of the tensor.
  const std::array<size_t, TViewRank>& Dimensions() const { return dimensions_; }

  /// Strides returns the strides of the tensor.
  const std::array<ssize_t, TViewRank>& Strides() const   { return strides_; }

  /// Size returns the data buffer size.
  size_t Size() const                                     { return size_; }

  /// Data returns a pointer to the data buffer of the view.
  pointer Data()
  {
    return reinterpret_cast<pointer>(pointer_cast<char*>(tensor_.Data())) + offset_;
  }

  const_pointer Data() const
  {
    return reinterpret_cast<pointer>(pointer_cast<char*>(tensor_.Data())) + offset_;
  }

  template <typename = void> requires requires (TTensor&& t) {t.Buffer();}
  auto Buffer()                                           { return tensor_.Buffer(); }

  template <typename = void> requires requires (TTensor&& t) {t.Buffer();}
  auto Buffer() const                                     { return tensor_.Buffer(); }

  /// Offset returns the offset in the buffer.
  size_t Offset() const                                   { return offset_; }

 private:
  TTensor&                        tensor_;
  std::array<size_t, TViewRank>   dimensions_;
  std::array<ssize_t, TViewRank>  strides_;
  size_t                          size_;
  size_t                          offset_;
};


namespace view {

/// Null defines the maximum size of a slice.
struct NullType {};
inline constexpr NullType Null{/*unspecified*/};

/// NewAxis inserts a new dimension to the view.
struct NewAxisType {};
inline constexpr NewAxisType&& NewAxis{/*unspecified*/};

/// Slice defines a slice starting from a start element to stop with an option step value.
struct Slice
{
  constexpr static ssize_t max_stop = std::numeric_limits<ssize_t>::max();
  constexpr Slice () : start_(0), stop_(max_stop), step_(1) {}
  constexpr Slice (ssize_t start) : start_(start), stop_(max_stop), step_(1) {}
  constexpr Slice (ssize_t start, ssize_t stop) : start_(start), stop_(stop), step_(1) {}
  constexpr Slice (ssize_t start, ssize_t stop, size_t step) : start_(start), stop_(stop), step_(step) {}
  constexpr Slice (ssize_t start, NullType&&) : start_(start), stop_(max_stop), step_(1) {}
  constexpr Slice (ssize_t start, NullType&&, size_t step) : start_(start), stop_(max_stop), step_(step) {}
  constexpr Slice (ssize_t start, const NullType&) : start_(start), stop_(max_stop), step_(1) {}
  constexpr Slice (ssize_t start, const NullType&, size_t step) : start_(start), stop_(max_stop), step_(step) {}

  const ssize_t start_;
  const ssize_t stop_;
  const size_t step_;
};


/// Extent defines a slice starting from a first element for a given length and an optional step value.
///
/// Extents are similar Slices except for expepcting a length argument instead of end index.
struct Extent : Slice
{
  constexpr Extent () : Slice() {}
  constexpr Extent (size_t extent) : Slice(0, extent) {}
  constexpr Extent (ssize_t start, size_t extent) : Slice(start, start + extent) {}
  constexpr Extent (ssize_t start, size_t extent, size_t step) : Slice(start, start + extent, step) {}
};

namespace {

template <typename T, typename... Ts>
struct accumulate
{
  constexpr static size_t count = (0 + ... + (std::is_base_of_v<T, std::remove_cvref_t<Ts>> ? 1 : 0));
};

} // end of namespace


/// View returns a view of the tensor with a subset of the tensor defined through slicing/indexing.
template <typename TTensor, typename... Ts>
inline auto View(TTensor& tensor, Ts&&... ts)
{
  // view rank:      number of slices (S) + 'remaining' tensor dims + new newaxiss (B)
  // remaining dims: tensor rank (R) - (number of arguments (N) - new newaxiss (B))
  constexpr size_t view_rank =
    accumulate<Slice, Ts...>::count +
    TTensor::rank -
    sizeof...(Ts) +
    accumulate<NewAxisType, Ts...>::count * 2;

  auto& tensor_dims = tensor.Dimensions();
  auto& tensor_strides = tensor.Strides();

  std::array<size_t, view_rank>  view_dims;
  std::array<ssize_t, view_rank> view_strides;

  size_t view_index = 0;
  size_t view_offset = 0;
  size_t tensor_index = 0;

  auto apply = [&] <typename T> (T&& value) {

    constexpr bool is_slice = std::is_base_of_v<Slice, std::decay_t<T>>;
    constexpr bool is_arithmetic = std::is_arithmetic_v<std::decay_t<T>>;
    constexpr bool is_newaxis = std::same_as<std::decay_t<T>, NewAxisType>;

    static_assert(is_slice || is_arithmetic || is_newaxis, "invalid argument to View()");

    ssize_t start = 0;
    size_t extent = 1;

    if constexpr (is_newaxis)
    {
      view_dims[view_index] = 1;
      view_strides[view_index] = 0;
      view_index++;
    }
    else
    {
      if constexpr (is_arithmetic)
      {
        start = value;
      }
      else if constexpr (is_slice)
      {
        // TODO: if extent <= 0 should return an empty (rank-0) tensor
        start = value.start_;
        extent = std::max(0L, value.stop_ - start);

        if (start < 0)
          start += tensor_dims[tensor_index];
        start = std::min<ssize_t>(start, (tensor_dims[tensor_index]));
        extent = std::min(extent, tensor_dims[tensor_index] - start);

        view_dims[view_index] = extent;
        view_strides[view_index] = tensor_strides[tensor_index] * value.step_;
        view_index++;
      }

      view_offset += start * tensor_strides[tensor_index];

      if (++tensor_index > tensor.Rank())
        throw std::runtime_error("index exceeds tensor rank");
    }
  };

  (apply(std::forward<Ts>(ts)),...);

  // copy remaining axes from tensor
  for (; view_index < view_rank; view_index++, tensor_index++)
  {
    view_dims[view_index] = tensor_dims[tensor_index];
    view_strides[view_index] = tensor_strides[tensor_index];
  }

  size_t view_size = get_block_size<typename TTensor::value_type>(view_dims, view_strides);
  return TensorView(tensor, view_dims, view_strides, view_size, view_offset);
}

// FIXME: offset should be in n-lemes
/// Reshape returns a view of the provide tensor with a new shape.
template <typename TTensor, size_t Rank>
inline auto Reshape(TTensor& tensor,
                    const std::array<size_t, Rank>& dimensions,
                    size_t offset = 0)
{
  using value_type = typename TTensor::value_type;
  auto strides = make_strides(dimensions);
  size_t size = get_array_size<value_type>(dimensions, strides);
  // assert orig-size >= size + offset
  return TensorView(tensor, dimensions, strides, size, offset);
}

template <typename TTensor, size_t Rank>
inline auto Reshape(TTensor& tensor,
                    const std::array<size_t, Rank>& dimensions,
                    const std::array<ssize_t, Rank>& strides,
                    size_t offset = 0)
{
  using value_type = typename TTensor::value_type;
  size_t size = get_array_size<value_type>(dimensions, strides);
  // assert orig-size >= size + offset
  return TensorView(tensor, dimensions, strides, size, offset);
}

} // end of namespace view
} // end of namespace libai

#endif  // LIBAI_TENSOR_TENSOR_VIEW_H
