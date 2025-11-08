//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef LIBAI_TENSOR_TENSOR_H
#define LIBAI_TENSOR_TENSOR_H

#include <functional>
#include <iterator>
#include <iostream>
#include <numeric>

#include "array.h"
#include "binary.h"
#include "comparison.h"
#include "concepts.h"
#include "device.h"
#include "function.h"
#include "iterator.h"
#include "generator.h"
#include "matmul.h"
#include "memory.h"
#include "tensor_parameters.h"
#include "tensor_view.h"
#include "unary.h"

// "CPU" is the default device for Tensors
#include "cpu/device.h"

namespace libai {

/// Tensor implements an "AI Tensor" that follows more typical AI implementations rather than
/// mathematical or physical definition.
///
/// @tparam T           Integral type
/// @tparam TRank       Rank of the tensor with 0: scalar, 1: vector, 2: matrix, etc.
///
/// Tensors define these member types and constexpr variables:
///
///   rank              TRank
///   value_type        T
///   memory_type       TMemory
///   pointer           Pointer type; depends on the implementation
///   const_pointer     Constant pointer type; depends on the implementation
///
/// Tensors also provide the following member methods:
///
///   constexpr size_t           Rank()
///   std::array<size_t, Rank>   Dimensions() const
///   std::array<ssize_t, Rank>  Strides() const
///   pointer                    Data()
///   const_pointer              Data() const
template <typename T, size_t TRank, typename TMemory>
class Tensor : public Array<T, TMemory>
{
  template <PrimitiveTensor P, size_t R> friend class TensorView;

  // helper to extract the template parameters for StaticMemory
  template <typename> struct mem_ext;
  template <size_t... Ns> struct mem_ext<libai::StaticMemory<Ns...>>
  {
    static constexpr std::array<size_t, sizeof...(Ns)> array{Ns...};
  };


 public:
  using value_type = T;
  using memory_type = TMemory;
  using pointer = value_type*;
  using reference = value_type&;
  using const_pointer = const value_type*;
  using const_reference = const value_type&;
  using array_type = Array<value_type, memory_type>;
  constexpr static size_t rank = TRank;


 public:
  Tensor() = default;

  //
  // Statically allocated memory
  //

  /// Constructor for a rank-0 tensor (scalar)
  Tensor(const value_type& init) : Array<value_type, memory_type>(1, init) {}
  Tensor(value_type&& init) : Array<value_type, memory_type>(1, init) {}
  Tensor(Uninitialized<value_type>) : Array<value_type, memory_type>(1) {}

  /// Constructor for a rank-1 tensor (vector) with static brace initialization.
  template <Arithmetic... Ts>
  Tensor(Ts&&... ts)
    : Array<std::common_type_t<Ts...>, StaticMemory<sizeof...(Ts)>>(std::to_array({std::forward<Ts>(ts)...})),
      dimensions_(std::to_array({sizeof...(Ts)})),
      strides_{1}
  {}

  /// Constructor for a rank-2 tensor (matrix) with static brace initialization.
  template <Arithmetic S, size_t... N>
  Tensor(S(&&... init)[N])
    : Array<value_type, memory_type>(get_array(std::move(init)...)),
      dimensions_(mem_ext<memory_type>::array),
      strides_{make_strides(dimensions_)}
  {}

  /// Constructor for a rank-3 tensor with static brace initialization.
  template <Arithmetic S, size_t... M, size_t... N>
  Tensor(S((&&... init)[M])[N])
    : Array<value_type, memory_type>(get_array(std::move(init)...)),
      dimensions_(mem_ext<memory_type>::array),
      strides_{make_strides(dimensions_)}
  {}

  //
  // Dynamically allocated memory
  //

  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated buffer without padding.
  explicit Tensor(size_t dimension, value_type init)
    : Array<value_type, memory_type>(dimension, init),
      dimensions_{dimension},
      strides_{make_strides(dimensions_)}
  {}

  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated uninitialized buffer.
  explicit Tensor(size_t dimension, Uninitialized<value_type>)
    : Array<value_type, memory_type>(dimension),
      dimensions_{dimension},
      strides_{make_strides(dimensions_)}
  {}

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated buffer and no padding.
  explicit Tensor(size_t dim_m, size_t dim_n, value_type init)
    : Array<value_type, memory_type>(dim_m * dim_n, init),
      dimensions_{dim_m, dim_n},
      strides_{make_strides(dimensions_)}
  {}

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated uninitialized buffer.
  explicit Tensor(size_t dim_m, size_t dim_n, Uninitialized<value_type>)
    : Array<value_type, memory_type>(dim_m * dim_n),
      dimensions_{dim_m, dim_n},
      strides_{make_strides(dimensions_)}
  {}

  /// Constructor for a rank-3 tensor (matrix) with a dynamically allocated buffer and no padding.
  explicit Tensor(size_t dim_m, size_t dim_n, size_t dim_u, value_type init)
    : Array<value_type, memory_type>(dim_m * dim_n * dim_u, init),
      dimensions_{dim_m, dim_n, dim_u},
      strides_{make_strides(dimensions_)}
  {}

  /// Constructor for a rank-3 tensor (matrix) with a dynamically allocated uninitialized buffer.
  explicit Tensor(size_t dim_m, size_t dim_n, size_t dim_u, Uninitialized<value_type>)
    : Array<value_type, memory_type>(dim_m * dim_n * dim_u),
      dimensions_{dim_m, dim_n, dim_u},
      strides_{make_strides(dimensions_)}
  {}


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit Tensor(std::initializer_list<size_t>&& dimensions, value_type init)
    : Array<value_type, memory_type>(std::accumulate(
          std::begin(dimensions), std::end(dimensions), 1, std::multiplies<size_t>()), init),
      dimensions_(get_array<size_t, TRank>(std::move(dimensions))),
      strides_{make_strides(dimensions_)}
  {}


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit Tensor(std::initializer_list<size_t>&& dimensions, Uninitialized<value_type>)
    : Array<value_type, memory_type>(std::accumulate(
          std::begin(dimensions), std::end(dimensions), 1, std::multiplies<size_t>())),
      dimensions_(get_array<size_t, TRank>(std::move(dimensions))),
      strides_{make_strides(dimensions_)}
  {}

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer with strides.
  explicit Tensor(std::initializer_list<size_t>&& dimensions,
                  std::initializer_list<ssize_t>&& strides,
                  value_type init)
    : Array<value_type, memory_type>(get_array_size<value_type>(dimensions, strides), dimensions, strides, init),
      dimensions_(get_array<size_t, TRank>(std::move(dimensions))),
      strides_(get_array<ssize_t, TRank>(std::move(strides)))
  {}

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer with strides.
  explicit Tensor(std::initializer_list<size_t>&& dimensions,
                  std::initializer_list<ssize_t>&& strides,
                  Uninitialized<value_type>)
    : Array<value_type, memory_type>(get_array_size<value_type>(dimensions, strides)),
      dimensions_(get_array<size_t, TRank>(std::move(dimensions))),
      strides_(get_array<ssize_t, TRank>(std::move(strides)))
  {}

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit Tensor(const size_t(&dimensions)[TRank], const ssize_t(&strides)[TRank], value_type init)
    : Array<value_type, memory_type>(dimensions, strides, init),
      dimensions_(get_array<size_t, TRank>(dimensions)),
      strides_(get_array<ssize_t, TRank>(strides))
  {}

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer
  explicit Tensor(const size_t(&dimensions)[TRank], const ssize_t(&strides)[TRank], Uninitialized<value_type>)
    : Array<value_type, memory_type>(dimensions, strides),
      dimensions_(get_array<size_t, TRank>(dimensions)),
      strides_(get_array<ssize_t, TRank>(strides))
  {}


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  explicit Tensor(const size_t(&dimensions)[TRank], value_type init)
    : Array<value_type, memory_type>(
        std::accumulate(std::begin(dimensions), std::end(dimensions), 1, std::multiplies<size_t>()), init),
      dimensions_(get_array<size_t, TRank>(dimensions)),
      strides_(make_strides(dimensions))
  {}

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer
  explicit Tensor(const size_t(&dimensions)[TRank], Uninitialized<value_type>)
    : Array<value_type, memory_type>(
        std::accumulate(std::begin(dimensions), std::end(dimensions), 1, std::multiplies<size_t>())),
      dimensions_(get_array<size_t, TRank>(dimensions)),
      strides_(make_strides(dimensions))
  {}


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer.
  explicit Tensor(std::array<size_t, TRank> dimensions, value_type init)
    : Array<value_type, memory_type>(
        std::accumulate(std::begin(dimensions), std::end(dimensions), 1, std::multiplies<size_t>()), init),
      dimensions_(dimensions),
      strides_(make_strides(dimensions))
  {}

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer with padding.
  explicit Tensor(std::array<size_t, TRank> dimensions,
                  std::array<ssize_t, TRank> strides,
                  value_type init)
    : Array<value_type, memory_type>(dimensions, strides, init),
      dimensions_{dimensions},
      strides_{strides}
  {}

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer.
  /// Note: assumes strides are type-aligned.
  explicit Tensor(std::array<size_t, TRank> dimensions, Uninitialized<value_type>)
    : Array<value_type, memory_type>(std::accumulate(
          std::begin(dimensions), std::end(dimensions), 1, std::multiplies<size_t>())),
      dimensions_{dimensions},
      strides_{make_strides(dimensions)}
  {}

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer with padding.
  explicit Tensor(std::array<size_t, TRank> dimensions,
                  std::array<ssize_t, TRank> strides,
                  Uninitialized<value_type>)
    : Array<value_type, memory_type>(get_array_size(dimensions, strides)),
      dimensions_{dimensions},
      strides_{strides}
  {}

  //
  // Copy/Move constructors
  //

  /// Copy constructor
  // TODO: "flatten" new array? check if already contiguous?
  Tensor(const Tensor& other)
    : Array<value_type, memory_type>(other, other.Dimensions(), other.Strides(), other.Strides()),
      dimensions_{other.Dimensions()},
      strides_{other.Strides()}
  {}

  // TODO: will be called when assigning StaticMemory to DeviceMemory
  template <AnyTensor TTensor>
  Tensor(const TTensor& other)
    : Array<value_type, memory_type>(other.Data(), other.Dimensions(), other.Strides(), other.Strides()),
      dimensions_{other.Dimensions()},
      strides_{other.Strides()}
  {}

  /// Move constructor
  Tensor(Tensor&& other)
    : Array<value_type, memory_type>(std::move(other)),
      dimensions_{std::move(other.dimensions_)},
      strides_{std::move(other.strides_)}
  {}

  // Constructors for converting from a tensor operator.
  template <AnyOperator TOperator>
  Tensor(TOperator&& functor) : Tensor{std::move(functor())} {};

  template <AnyOperator TOperator>
  Tensor(const TOperator& functor) : Tensor{functor()} {};


  /// Assign operator
  template <PrimitiveTensor TTensor>
  Tensor& operator=(const TTensor& other)
  {
    dimensions_ = other.Dimensions();
    strides_ = other.Strides();
    if (array_type::Size() != other.Size())
      array_type::Realloc(other.Size()); // FIXME it's n-elems..
    Copy(*this, other);
    return *this;
  }

  /// Move-assign is only supported from the same type
  Tensor& operator=(Tensor&& other)
  {
    dimensions_ = other.Dimensions();
    strides_ = other.Strides();
    array_type::operator=(std::move(other));
    return *this;
  }

  /// Operator assign
  template <AnyOperator TOperator>
  Tensor& operator=(TOperator&& oper)
  {
    return operator=(std::forward<TOperator>(oper)());
  }

  template <AnyOperator TOperator>
  Tensor& operator+=(TOperator&& oper)
  {
    return operator=(Add(*this, std::forward<TOperator>(oper)()));
  }


  /// View returns a view of the proivded tensor.
  template <typename... Ts>
  auto View(Ts&&... slices)
  {
    return view::View(*this, std::forward<Ts>(slices)...);
  }

  template <typename... Ts>
  auto View(Ts&&... slices) const
  {
    return view::View(*this, std::forward<Ts>(slices)...);
  }

  /// Reshape returns a view of the tensor with a different shape (axes, dimensions, strides)
  /// of the underlying array.
  template <size_t TViewRank>
  auto Reshape(const std::array<size_t, TViewRank>& dimensions,
               const std::array<ssize_t, TViewRank>& strides)
  {
    return view::Reshape(*this, std::move(dimensions), std::move(strides));
  }

  template <size_t TViewRank>
  auto Reshape(const std::array<size_t, TViewRank>& dimensions,
               const std::array<ssize_t, TViewRank>& strides) const
  {
    return view::Reshape(*this, std::move(dimensions), std::move(strides));
  }

  template <size_t TViewRank>
  auto Reshape(const std::array<size_t, TViewRank>& dimensions)
  {
    return view::Reshape(*this, std::move(dimensions));
  }

  template <size_t TViewRank>
  auto Reshape(const std::array<size_t, TViewRank>& dimensions) const
  {
    return view::Reshape(*this, std::move(dimensions));
  }


  /// begin returns an iterator for the begin of the Tensor array
  auto begin()                        { return details::Iterator(*this); }

  // TODO: return "sentinel"!!
  /// end returns the sentinel for the end of the Tensor array
  auto end()                          { return details::Iterator(*this, Dimensions()); }

  /// begin returns an iterator for the begin of the Tensor array
  auto begin() const                  { return details::ConstIterator(*this); }

  /// end returns the sentinel for the end of the Tensor array
  auto end() const                    { return details::ConstIterator(*this, Dimensions()); }


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return TRank; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, TRank>& Dimensions() const     { return dimensions_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, TRank>& Strides() const       { return strides_; }

  /// Offset returns the offset in the buffer.
  size_t Offset() const                                   { return 0UL; }

 private:
  std::array<size_t, TRank>         dimensions_;
  std::array<ssize_t, TRank>        strides_;
};


/// Tensor<T, Rank, MemoryMapped> is a tensor for an externally managed buffer
template <typename T, size_t TRank>
class Tensor<T, TRank, MemoryMapped>
{
 public:
  using value_type = T;
  using memory_type = MemoryMapped;
  using pointer = const value_type*;
  using reference = const value_type&;
  using const_pointer = const value_type*;
  using const_reference = const value_type&;
  using array_type = Array<value_type, memory_type>;
  constexpr static size_t rank = TRank;

  explicit Tensor() {}

  // FIXME: description for buffer with byte size!! or should this be n-elems??
  explicit Tensor(const size_t(&& dimensions)[TRank], const std::tuple<pointer, size_t>& array)
    : dimensions_(std::to_array(dimensions)),
      strides_{make_strides(dimensions_)},
      size_(dimensions_[0] * strides_[0]),
      data_(std::get<0>(array))
  {
    if (size_ > std::get<1>(array))
      throw std::runtime_error("dimensions exceed allotted size: " + std::to_string(size_) + " > " +
          std::to_string(std::get<1>(array)));
    if (size_ == 0UL)
      throw std::runtime_error("attempting to create a zero-size memory mapped tensor");
  }

  explicit Tensor(const std::array<size_t, TRank>& dimensions, const std::tuple<pointer, size_t>& array)
    : dimensions_(dimensions),
      strides_{make_strides(dimensions_)},
      size_(dimensions_[0] * strides_[0]),
      data_(std::get<0>(array))
  {
    if (size_ > std::get<1>(array))
      throw std::runtime_error("dimensions exceed allotted size: " + std::to_string(size_) + " > " +
          std::to_string(std::get<1>(array)));
    if (size_ == 0UL)
      throw std::runtime_error("attempting to create a zero-size memory mapped tensor");
  }


  Tensor& operator=(Tensor&& other)
  {
    dimensions_ = other.dimensions_;
    strides_ = other.strides_;
    size_ = other.size_;
    data_ = other.data_;
    return *this;
  }

  Tensor& operator=(Tensor& other)
  {
    dimensions_ = other.dimensions_;
    strides_ = other.strides_;
    size_ = other.size_;
    data_ = other.data_;
    return *this;
  }

  Tensor(Tensor& other)
    : dimensions_(other.dimensions_),
      strides_(other.strides_),
      size_(other.size_),
      data_(other.data_)
  {
  }

  Tensor(Tensor&& other)
    : dimensions_(other.dimensions_),
      strides_(other.strides_),
      size_(other.size_),
      data_(other.data_)
  {
    other.data_ = nullptr;
  }


  /// View returns a view of the proivded tensor.
  template <typename... Ts>
  auto View(Ts&&... slices)
  {
    return view::View(*this, std::forward<Ts>(slices)...);
  }

  template <typename... Ts>
  auto View(Ts&&... slices) const
  {
    return view::View(*this, std::forward<Ts>(slices)...);
  }

  /// Reshape returns a view of the tensor with a different shape (axes, dimensions, strides)
  /// of the underlying array.
  template <size_t TViewRank>
  auto Reshape(const std::array<size_t, TViewRank>& dimensions,
               const std::array<ssize_t, TViewRank>& strides)
  {
    return view::Reshape(*this, std::to_array(dimensions), std::to_array(strides));
  }

  template <size_t TViewRank>
  auto Reshape(const std::array<size_t, TViewRank>& dimensions,
               const std::array<ssize_t, TViewRank>& strides) const
  {
    return view::Reshape(*this, std::to_array(dimensions), std::to_array(strides));
  }


  /// begin returns an iterator for the begin of the Tensor array
  auto begin() const                  { return details::ConstIterator(*this); }

  /// end returns the sentinel for the end of the Tensor array
  auto end() const                    { return details::ConstIterator(*this, Dimensions()); }


  /// Rank returns the rank of the tensor.
  constexpr static size_t Rank()                          { return TRank; }

  /// Dimensions returns the dimensions of the tensor.
  const std::array<size_t, TRank>& Dimensions() const     { return dimensions_; }

  /// Strides returns the strides of the tensor.
  const std::array<ssize_t, TRank>& Strides() const       { return strides_; }

  /// Size returns the data buffer size.
  size_t Size() const                                     { return size_; }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return data_; }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return data_; }

  /// Offset returns the offset in the buffer.
  size_t Offset() const                                   { return 0UL; }

 private:
  std::array<size_t, TRank>   dimensions_;
  std::array<ssize_t, TRank>  strides_;
  size_t                      size_;
  pointer                     data_;
};


//
// CTAD rules
//

// Tensor rules for rank-0 tensors

// Tensor{T} -> Rank-0 tensor with a static/local array
template <Arithmetic T>
explicit Tensor(T) -> Tensor<T, 0, Scalar>;

// Tensor{Uninitailzied<T>} -> Rank-0 tensor with a static/local array
template <Arithmetic T>
explicit Tensor(Uninitialized<T>) -> Tensor<T, 0, Scalar>;

// Tensor with Static Allocator - Brace-initializer List

// Tensor{Ts...} -> Rank-1 tensor with a static/local array (brace-initializer).
template <Arithmetic... Ts>
Tensor(Ts...) -> Tensor<std::common_type_t<Ts...>, 1, StaticMemory<sizeof...(Ts)>>;

// Tensor{{...},...} -> Rank-2 tensor with a static/local array (brace-initializer).
template <Arithmetic T, size_t... N>
Tensor(T(&&... l)[N]) -> Tensor<T, 2, StaticMemory<sizeof...(N), std::max({N...})>>;

// Tensor{{{...},...},...} -> Rank-3 tensor with a static/local array (brace-initializer).
template <Arithmetic T, size_t... M, size_t... N>
Tensor(T(&&... l)[M][N]) -> Tensor<T, 3, StaticMemory<sizeof...(M), std::max({M...}), std::max({N...})>>;

// Tensor rules for allocating dynamic device memory with dimensions provded as arguments

// TODO: These rules are currently ignored by gcc as integers are not deduced from an alias definition.
//       This might be by design or because of an issue with gcc. See:
//        - https://stackoverflow.com/questions/64939408/how-to-write-deduction-guidelines-for-aliases-of-aggregate-templates
//        - https://stackoverflow.com/questions/41008092/class-template-argument-deduction-not-working-with-alias-template

#if 0

// Tensor(uint,T) -> Rank-1 tensor with a dynamically allocated buffer.
template <Arithmetic T, typename Dev = device::CPU>
explicit Tensor(size_t, T) -> Tensor<T, 1, DeviceMemory<Dev>>;

// Tensor(uint, Uninitialized<T>) -> Rank-1 tensor with a dynamically allocated uninitialized buffer.
template <Arithmetic T, typename Dev = device::CPU>
explicit Tensor(size_t, Uninitialized<T>) -> Tensor<T, 1, DeviceMemory<Dev>>;

// Tensor(uint, uint, T) -> Rank-2 tensor with a dynamically allocated buffer.
template <Arithmetic T, typename Dev = device::CPU>
explicit Tensor(size_t, size_t, T) -> Tensor<T, 2, DeviceMemory<Dev>>;

// Tensor(uint, Uninitialized<T>) -> Rank-2 tensor with a dynamically allocated uninitialized buffer.
template <Arithmetic T, typename Dev = device::CPU>
explicit Tensor(size_t, size_t, Uninitialized<T>) -> Tensor<T, 2, DeviceMemory<Dev>>;

// Tensor(size_t, size_t, size_t, Uninitialized<T>) -> Rank-3 tensor with a dynamically allocated uninitialized buffer.
template <Arithmetic T, typename Dev = device::CPU>
explicit Tensor(size_t, size_t, size_t, T) -> Tensor<T, 3, DeviceMemory<Dev>>;

// Tensor(size_t, size_t, size_t, Uninitialized<T>) -> Rank-3 tensor with a dynamically allocated uninitialized buffer.
template <Arithmetic T, typename Dev = device::CPU>
explicit Tensor(size_t, size_t, size_t, Uninitialized<T>) -> Tensor<T, 3, DeviceMemory<Dev>>;

#endif

// Tensor rules for allocation dynamic device memory with dimensions and optional strides provided as arrays

// Tensor(&[], &[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <Arithmetic T, size_t N, typename Dev = device::CPU>
explicit Tensor(const size_t(&)[N], const ssize_t(&)[N], T) -> Tensor<T, N, DeviceMemory<Dev>>;

// Tensor(&[], &[], Uninitialized<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <Arithmetic T, size_t N, typename Dev = device::CPU>
explicit Tensor(const size_t(&)[N], const ssize_t(&)[N], Uninitialized<T>) -> Tensor<T, N, DeviceMemory<Dev>>;

// Tensor(&&[], &&[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <Arithmetic T, size_t N, typename Dev = device::CPU>
explicit Tensor(size_t(&&)[N], ssize_t(&&)[N], T) -> Tensor<T, N, DeviceMemory<Dev>>;

// Tensor(&&[], &&[]) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <Arithmetic T, size_t N, typename Dev = device::CPU>
explicit Tensor(size_t(&&)[N], ssize_t(&&)[N], Uninitialized<T>) -> Tensor<T, N, DeviceMemory<Dev>>;

// Tensor(&[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <Arithmetic T, size_t N, typename Dev = device::CPU>
explicit Tensor(const size_t(&)[N], T) -> Tensor<T, N, DeviceMemory<Dev>>;

// Tensor(&[], Uninitialized<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <Arithmetic T, size_t N, typename Dev = device::CPU>
explicit Tensor(const size_t(&)[N], Uninitialized<T>) -> Tensor<T, N, DeviceMemory<Dev>>;

// Tensor(&&[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <Arithmetic T, size_t N, typename Dev = device::CPU>
explicit Tensor(const size_t(&&)[N], T) -> Tensor<T, N, DeviceMemory<Dev>>;

// Tensor(&&[], Uninitialized<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <Arithmetic T, size_t N, typename Dev = device::CPU>
explicit Tensor(const size_t(&&)[N], Uninitialized<T>) -> Tensor<T, N, DeviceMemory<Dev>>;

// Tensor(array, T)
template <Arithmetic T, size_t N, typename Dev = device::CPU>
Tensor(std::array<size_t, N>, T) -> Tensor<T, N, DeviceMemory<Dev>>;

// Tensor(array, array, T)
template <Arithmetic T, size_t N, typename Dev = device::CPU>
explicit Tensor(std::array<size_t, N>, std::array<ssize_t, N>, T) -> Tensor<T, N, DeviceMemory<Dev>>;

// Tensor(array, Uninitialized<T>)
template <Arithmetic T, size_t N, typename Dev = device::CPU>
explicit Tensor(std::array<size_t, N>, Uninitialized<T>) -> Tensor<T, N, DeviceMemory<Dev>>;

// Tensor(array, array, Uninitialized<T>)
template <Arithmetic T, size_t N, typename Dev = device::CPU>
explicit Tensor(std::array<size_t, N>, std::array<ssize_t, N>, Uninitialized<T>) -> Tensor<T, N, DeviceMemory<Dev>>;

// Tensor copy constructor
template <AnyTensor TTensor, typename Dev = device::CPU>
Tensor(const TTensor& other) -> Tensor<typename TTensor::value_type, TTensor::rank, DeviceMemory<Dev>>;

// Tensor rules for tensor view argument

template <typename TTensor, size_t TRank, typename Dev = device::CPU>
Tensor(TensorView<TTensor, TRank>&&) -> Tensor<typename TTensor::value_type, TRank, DeviceMemory<Dev>>;
template <typename TTensor, size_t TRank, typename Dev = device::CPU>
Tensor(const TensorView<TTensor, TRank>&) -> Tensor<typename TTensor::value_type, TRank, DeviceMemory<Dev>>;

// Tensor rules for operator arguments

template <libai::AnyOperator TOperator, typename Dev = device::CPU>
Tensor(TOperator&&) -> Tensor<typename TOperator::value_type, TOperator::rank, libai::DeviceMemory<Dev>>;

template <libai::AnyOperator TOperator, typename Dev = device::CPU>
Tensor(const TOperator&) -> Tensor<typename TOperator::value_type, TOperator::rank, libai::DeviceMemory<Dev>>;

// Tensor rules for memory-mapped arguments

template <Arithmetic T, size_t N>
explicit Tensor(const size_t(&)[N], const std::tuple<T*, size_t>&) -> Tensor<T, N, MemoryMapped>;
template <Arithmetic T, size_t N>
explicit Tensor(const std::array<size_t, N>&, const std::tuple<T*, size_t>&) -> Tensor<T, N, libai::MemoryMapped>;

//
// Arithmentic operator overloading
//

// operator+ (TensorType, TensorType)
template <TensorConvertible TTensor1, TensorConvertible TTensor2>
auto operator+(TTensor1&& tensor1, TTensor2&& tensor2)
{
  return Add(std::forward<TTensor1>(tensor1), std::forward<TTensor2>(tensor2));
}

// operator* (TensorType, TensorType) -> Mul, requires same rank
template <TensorConvertible TTensor1, TensorConvertible TTensor2>
auto operator*(TTensor1&& tensor1, TTensor2&& tensor2)
requires (std::decay_t<TTensor1>::rank == std::decay_t<TTensor2>::rank)
{
  return Mul(std::forward<TTensor1>(tensor1), std::forward<TTensor2>(tensor2));
}

// operator* (TensorType, TensorType) -> Scale, if one Tensor has rank0
template <TensorConvertible TTensor1, TensorConvertible TTensor2>
auto operator*(TTensor1&& tensor1, TTensor2&& tensor2)
requires (std::decay_t<TTensor1>::rank == 0 || std::decay_t<TTensor2>::rank == 0)
{
  return Mul(std::forward<TTensor1>(tensor1), std::forward<TTensor2>(tensor2));
}

// operator* (TensorType, arithmetic)
template <TensorConvertible TTensor, Arithmetic T>
auto operator*(TTensor&& tensor, T scalar)
{
  return Mul(std::forward<TTensor>(tensor), scalar);
}

// operator* (arithmetic, TensorType)
template <Arithmetic T, TensorConvertible TTensor>
auto operator*(T scalar, TTensor&& tensor)
{
  return Mul(scalar, std::forward<TTensor>(tensor));
}

// operator/ (TensorType, TensorType)
template <TensorConvertible TTensor1, TensorConvertible TTensor2>
auto operator/(TTensor1&& tensor1, TTensor2&& tensor2)
{
  return Div(std::forward<TTensor1>(tensor1), std::forward<TTensor2>(tensor2));
}

// operator/ (TensorType, arithmetic)
template <TensorConvertible TTensor, Arithmetic T>
auto operator/(TTensor&& tensor, T scalar)
{
  return Div(std::forward<TTensor>(tensor), scalar);
}

} // end of namespace libai

/// operator<<(TENSOR) overloads the output operator for tensors.
std::ostream& operator<<(std::ostream& os, const libai::AnyTensor auto& tensor)
{
  using value_type = typename std::remove_reference_t<decltype(tensor)>::value_type;
  size_t rank = tensor.Rank();

  auto dimensions = tensor.Dimensions();
  auto strides = tensor.Strides();

  std::function<void(int, const value_type*)> print;
  print = [&os, &dimensions, &strides, &print, &rank](size_t index, const value_type* ptr) {
    os << "{ ";
    if (index < rank - 1)
    {
      for (size_t i = dimensions[index]; i > 0; i--)
      {
        print(index + 1, ptr);
        if (i != 1)
          os << ", ";
        else
          os << " }";
        ptr += strides[index];
      }
    }
    else
    {
      for (size_t i = dimensions[rank-1]; i > 0; i--)
      {
        os << *ptr;
        if (i != 1)
          os << ", ";
        else
          os << " }";
        ptr += strides[rank-1];
      }
    }
  };
  const value_type* ptr = reinterpret_cast<const value_type*>(tensor.Data());
  if (rank > 0)
    print(0, ptr);
  else
    os << "{ " << *ptr << " }";

  os << std::flush;

  return os;
}

#endif  // LIBAI_TENSOR_TENSOR_H
