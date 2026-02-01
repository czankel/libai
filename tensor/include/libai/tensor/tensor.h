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
#include "allocator.h"
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
/// @tparam NRank       Rank of the tensor with 0: scalar, 1: vector, 2: matrix, etc.
///
/// Tensors define these member types and constexpr variables:
///
///   rank              NRank
///   value_type        T
///   allocator_type    TAllocator
///   pointer           Pointer type; depends on the implementation
///   const_pointer     Constant pointer type; depends on the implementation
///
/// Tensors also provide the following member methods:
///
///   constexpr size_t           Rank()
///   std::array<size_t, rank>   Dimensions() const
///   std::array<ssize_t, rank>  Strides() const
///   pointer                    Data()
///   const_pointer              Data() const
template <typename T, size_t NRank,
          typename TDevice = device::CPU,
          typename TAllocator = typename TDevice::template allocator_type<T>>
class Tensor
{
  template <PrimitiveTensor, size_t> friend class TensorView;
  template <typename, size_t, typename, typename> friend class Tensor;

  // helper to extract the template parameters for StaticResource
  template <typename> struct mem_ext;
  template <size_t... Ns> struct mem_ext<libai::StaticResource<Ns...>>
  {
    static constexpr std::array<size_t, sizeof...(Ns)> array{Ns...};
  };


 public:
  using value_type = T;
  using pointer = value_type*;
  using reference = value_type&;
  using const_pointer = const value_type*;
  using const_reference = const value_type&;
  using device_type = TDevice;
  using allocator_type = TAllocator;
  using array_type = Array<value_type, NRank, allocator_type>;
  constexpr static size_t rank = NRank;


 public:
  Tensor() = default;

  //
  // Statically allocated memory
  //

  /// Constructor for a rank-0 tensor (scalar)
  explicit Tensor(const value_type& init) : array_(1, init) {}
  explicit Tensor(value_type&& init) : array_(1, init) {}
  explicit Tensor(std::type_identity<value_type>) : array_(1) {}

  /// Constructor for a rank-1 tensor (vector) with static brace initialization.
  template <Arithmetic... Ts>
  explicit Tensor(Ts&&... ts)
    : array_(std::to_array({sizeof...(Ts)}),
             std::to_array({std::forward<Ts>(ts)...})),
      dimensions_(std::to_array({sizeof...(Ts)})),
      strides_{1}
  {}

  /// Constructor for a rank-2 tensor (matrix) with static brace initialization.
  /// allow implicit conversion, e.g. from brace
  template <Arithmetic S, size_t... N>
  Tensor(S(&&... init)[N])
    : array_(mem_ext<allocator_type>::array, get_array(std::move(init)...)),
      dimensions_(mem_ext<allocator_type>::array),
      strides_{make_strides(dimensions_)}
  {}

  /// Constructor for a rank-3 tensor with static brace initialization.
  template <Arithmetic S, size_t... M, size_t... N>
  Tensor(S((&&... init)[M])[N])
    : array_(mem_ext<allocator_type>::array, get_array(std::move(init)...)),
      dimensions_(mem_ext<allocator_type>::array),
      strides_{make_strides(dimensions_)}
  {}

  //
  // Dynamically allocated memory
  //

  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated buffer without padding.
  Tensor(size_t dimension, value_type init)
    : array_(dimension, init),
      dimensions_{dimension},
      strides_{1}
  {}

  /// Constructor for a rank-1 tensor (vector) with a dynamically allocated uninitialized buffer.
  Tensor(size_t dimension, std::type_identity<value_type>)
    : array_(dimension, std::type_identity<value_type>{}),
      dimensions_{dimension},
      strides_{make_strides(dimensions_)}
  {}

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated buffer and no padding.
  Tensor(size_t dim_m, size_t dim_n, value_type init)
    : array_(dim_m * dim_n, init),
      dimensions_{dim_m, dim_n},
      strides_{make_strides(dimensions_)}
  {}

  /// Constructor for a rank-2 tensor (matrix) with a dynamically allocated uninitialized buffer.
  Tensor(size_t dim_m, size_t dim_n, std::type_identity<value_type>)
    : array_(dim_m * dim_n, std::type_identity<value_type>{}),
      dimensions_{dim_m, dim_n},
      strides_{make_strides(dimensions_)}
  {}

  /// Constructor for a rank-3 tensor (matrix) with a dynamically allocated buffer and no padding.
  Tensor(size_t dim_m, size_t dim_n, size_t dim_u, value_type init)
    : array_(dim_m * dim_n * dim_u, init),
      dimensions_{dim_m, dim_n, dim_u},
      strides_{make_strides(dimensions_)}
  {}

  /// Constructor for a rank-3 tensor (matrix) with a dynamically allocated uninitialized buffer.
  Tensor(size_t dim_m, size_t dim_n, size_t dim_u, std::type_identity<value_type>)
    : array_(dim_m * dim_n * dim_u, std::type_identity<value_type>{}),
      dimensions_{dim_m, dim_n, dim_u},
      strides_{make_strides(dimensions_)}
  {}


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  Tensor(std::initializer_list<size_t>&& dimensions, value_type init)
    : array_(std::accumulate(
          std::begin(dimensions), std::end(dimensions), 1, std::multiplies<size_t>()), init),
      dimensions_(get_array<size_t, rank>(std::move(dimensions))),
      strides_{make_strides(dimensions_)}
  {}


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  Tensor(std::initializer_list<size_t>&& dimensions, std::type_identity<value_type>)
    : array_(std::accumulate(
               std::begin(dimensions), std::end(dimensions), 1, std::multiplies<size_t>()),
             std::type_identity<value_type>{}),
      dimensions_(get_array<size_t, rank>(std::move(dimensions))),
      strides_{make_strides(dimensions_)}
  {}

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer with strides.
  Tensor(std::initializer_list<size_t>&& dimensions,
         std::initializer_list<ssize_t>&& strides,
         value_type init)
    : array_(get_array_size(dimensions, strides), dimensions, strides, init),
      dimensions_(get_array<size_t, rank>(std::move(dimensions))),
      strides_(get_array<ssize_t, rank>(std::move(strides)))
  {}

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer with strides.
  Tensor(std::initializer_list<size_t>&& dimensions,
         std::initializer_list<ssize_t>&& strides,
         std::type_identity<value_type>)
    : array_(get_array_size(dimensions, strides), std::type_identity<value_type>{}),
      dimensions_(get_array<size_t, rank>(std::move(dimensions))),
      strides_(get_array<ssize_t, rank>(std::move(strides)))
  {}

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  Tensor(const size_t(&dimensions)[rank], const ssize_t(&strides)[rank], value_type init)
    : array_(dimensions, strides, init),
      dimensions_(get_array<size_t, rank>(dimensions)),
      strides_(get_array<ssize_t, rank>(strides))
  {}

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer
  Tensor(const size_t(&dimensions)[rank], const ssize_t(&strides)[rank], std::type_identity<value_type>)
    : array_(dimensions, strides, std::type_identity<value_type>{}),
      dimensions_(get_array<size_t, rank>(dimensions)),
      strides_(get_array<ssize_t, rank>(strides))
  {}


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer
  Tensor(const size_t(&dimensions)[rank], value_type init)
    : array_(
        std::accumulate(std::begin(dimensions), std::end(dimensions), 1, std::multiplies<size_t>()), init),
      dimensions_(get_array<size_t, rank>(dimensions)),
      strides_(make_strides(dimensions))
  {}

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer
  Tensor(const size_t(&dimensions)[rank], std::type_identity<value_type>)
    : array_(
        std::accumulate(std::begin(dimensions), std::end(dimensions), 1, std::multiplies<size_t>()), std::type_identity<value_type>{}),

      dimensions_(get_array<size_t, rank>(dimensions)),
      strides_(make_strides(dimensions))
  {}


  /// Constructor for any rank tensor with a dynamically allocated initialized buffer.
  Tensor(std::array<size_t, rank> dimensions, value_type init)
    : array_(
        std::accumulate(std::begin(dimensions), std::end(dimensions), 1, std::multiplies<size_t>()), init),
      dimensions_(dimensions),
      strides_(make_strides(dimensions))
  {}

  /// Constructor for any rank tensor with a dynamically allocated initialized buffer with padding.
  Tensor(std::array<size_t, rank> dimensions,
         std::array<ssize_t, rank> strides,
         value_type init)
    : array_(dimensions, strides, init),
      dimensions_{dimensions},
      strides_{strides}
  {}

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer.
  /// Note: assumes strides are type-aligned.
  Tensor(std::array<size_t, rank> dimensions, std::type_identity<value_type>)
    : array_(std::accumulate(
          std::begin(dimensions), std::end(dimensions), 1, std::multiplies<size_t>()), std::type_identity<value_type>{}),
      dimensions_{dimensions},
      strides_{make_strides(dimensions)}
  {}

  /// Constructor for any rank tensor with a dynamically allocated uninitialized buffer with padding.
  Tensor(std::array<size_t, rank> dimensions,
         std::array<ssize_t, rank> strides,
         std::type_identity<value_type>)
    : array_(get_array_size(dimensions, strides), std::type_identity<value_type>{}),
      dimensions_{dimensions},
      strides_{strides}
  {}

  /// Constructor for memory mapped arrays
  Tensor(const std::array<size_t, rank>& dimensions, const std::tuple<pointer, size_t>& mmap)
    : array_(std::get<0>(mmap), dimensions),
      dimensions_{dimensions},
      strides_{make_strides(dimensions_)}
  {
    if (get_array_size(dimensions_, strides_) > std::get<1>(mmap))
      throw std::runtime_error("tensor dimensions and strides exceed mmaped area");
  }

  Tensor(const size_t(& dimensions)[rank], const std::tuple<T*, size_t>& mmap)
    : array_(std::get<0>(mmap), std::to_array(dimensions)),
      dimensions_(std::to_array(dimensions)),
      strides_{make_strides(dimensions_)}
  {
    if (get_array_size(dimensions_, strides_) > std::get<1>(mmap))
      throw std::runtime_error("tensor dimensions and strides exceed mmaped area");
  }

  /// Constructor from a Tensor of a different value and array type.
  template <AnyTensor TTensor>
  Tensor(const TTensor& other)
    : array_(other.array_),
      dimensions_{other.Dimensions()},
      strides_{other.Strides()}
  {}

  // FIXNE: add note that we don't pass TensorView as const ref so we can modify it.. provide both?
  template <AnyTensor TTensor>
  Tensor(TensorView<TTensor, rank>& view)
    : array_(view.array_),
      dimensions_{view.Dimensions()},
      strides_{view.Strides()}
  {}

  template <AnyTensor TTensor>
  Tensor(TensorView<TTensor, rank>&& view)
    : array_(view.array_),
      dimensions_{view.Dimensions()},
      strides_{view.Strides()}
  {}

  // Constructors for converting from a tensor operator.
  // Allow implicit conversions
  template <AnyOperator TOperator>
  Tensor(TOperator&& functor) : Tensor{std::move(functor())} {};

  template <AnyOperator TOperator>
  Tensor(const TOperator& functor) : Tensor{functor()} {};

  //
  // Copy/Move constructors
  //

  /// Copy constructor
  // TODO: "flatten" new array? check if already contiguous?
  Tensor(const Tensor& other)
    : array_(other.array_),
      dimensions_{other.Dimensions()},
      strides_{other.Strides()}
  {}

  /// Move constructor
  Tensor(Tensor&& other)
    : array_(std::move(other.array_)),
      dimensions_{std::move(other.dimensions_)},
      strides_{std::move(other.strides_)}
  {}

  //
  // Assign operators
  //

  template <PrimitiveTensor TTensor>
  Tensor& operator=(const TTensor& other)
  {
    dimensions_ = other.Dimensions();
    strides_ = other.Strides();
    array_ = other.array_;
    return *this;
  }

  Tensor& operator=(Tensor&& other)
  {
    dimensions_ = other.Dimensions();
    strides_ = other.Strides();
    array_ = std::move(other.array_);
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

  // TODO: investigate if *this should be rvalue, etc.
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
  constexpr static size_t Rank()                          { return rank; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, rank>& Dimensions() const      { return dimensions_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, rank>& Strides() const        { return strides_; }

  /// Offset returns the offset in the buffer.
  size_t Offset() const                                   { return 0UL; }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return get_array_size(dimensions_, strides_); }

  /// Data returns a pointer to the data buffer.
  auto Data()                                             { return array_.Data(); }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return array_.Data(); }

  // TODO: temporary addition to support Metal buffers

  // Buffer returns the MTL buffer - internal use only
  auto Buffer()                                           { return array_.Buffer(); }

  // Buffer returns the MTL buffer - internal use only
  auto Buffer() const                                     { return array_.Buffer(); }


 private:
  Array<value_type, rank, allocator_type> array_;
  std::array<size_t, rank>                dimensions_;
  std::array<ssize_t, rank>               strides_;
};


//
// CTAD rules
//

// Tensor rules for rank-0 tensors

// Tensor{T} -> Rank-0 tensor with a static/local array
template <Arithmetic T>
explicit Tensor(T) -> Tensor<T, 0, device::CPU, Scalar>;

// Tensor{std::type_identity<T>} -> Rank-0 tensor with a static/local array
template <Arithmetic T>
explicit Tensor(std::type_identity<T>) -> Tensor<T, 0, device::CPU, Scalar>;

// Tensor with Static Allocator - Brace-initializer List

// Tensor{Ts...} -> Rank-1 tensor with a static/local array (brace-initializer).
template <Arithmetic... Ts>
Tensor(Ts...) -> Tensor<std::common_type_t<Ts...>, 1, device::CPU, StaticResource<sizeof...(Ts)>>;

// Tensor{{...},...} -> Rank-2 tensor with a static/local array (brace-initializer).
template <Arithmetic T, size_t... N>
Tensor(T(&&... l)[N]) -> Tensor<T, 2, device::CPU, StaticResource<sizeof...(N), std::max({N...})>>;

// Tensor{{{...},...},...} -> Rank-3 tensor with a static/local array (brace-initializer).
template <Arithmetic T, size_t... M, size_t... N>
Tensor(T(&&... l)[M][N])
  -> Tensor<T, 3, device::CPU, StaticResource<sizeof...(M), std::max({M...}), std::max({N...})>>;

// Tensor rules for allocating dynamic device memory with dimensions provded as arguments

// TODO: These rules are currently ignored by gcc as integers are not deduced from an alias definition.
//       This might be by design or because of an issue with gcc. See:
//        - https://stackoverflow.com/questions/64939408/how-to-write-deduction-guidelines-for-aliases-of-aggregate-templates
//        - https://stackoverflow.com/questions/41008092/class-template-argument-deduction-not-working-with-alias-template

#if 0

// Tensor(uint,T) -> Rank-1 tensor with a dynamically allocated buffer.
template <Arithmetic T, typename TDevice = device::CPU>
Tensor(size_t, T) -> Tensor<T, 1, TDevice, typename TDevice::template allocator_type<T>>;

// Tensor(uint, std::type_identity<T>) -> Rank-1 tensor with a dynamically allocated uninitialized buffer.
template <Arithmetic T, typename TDevice = device::CPU>
Tensor(size_t, std::type_identity<T>) -> Tensor<T, 1, TDevice, typename TDevice::template allocator_type<T>>;

// Tensor(uint, uint, T) -> Rank-2 tensor with a dynamically allocated buffer.
template <Arithmetic T, typename TDevice = device::CPU>
Tensor(size_t, size_t, T) -> Tensor<T, 2, TDevice, typename TDevice::template allocator_type<T>>;

// Tensor(uint, std::type_identity<T>) -> Rank-2 tensor with a dynamically allocated uninitialized buffer.
template <Arithmetic T, typename TDevice = device::CPU>
Tensor(size_t, size_t, std::type_identity<T>) -> Tensor<T, 2, TDevice, typename TDevice::template allocator_type<T>>;

// Tensor(size_t, size_t, size_t, std::type_identity<T>) -> Rank-3 tensor with a dynamically allocated uninitialized buffer.
template <Arithmetic T, typename TDevice = device::CPU>
Tensor(size_t, size_t, size_t, T) -> Tensor<T, 3, TDevice, typename TDevice::template allocator_type<T>>;

// Tensor(size_t, size_t, size_t, std::type_identity<T>) -> Rank-3 tensor with a dynamically allocated uninitialized buffer.
template <Arithmetic T, typename TDevice = device::CPU>
Tensor(size_t, size_t, size_t, std::type_identity<T>) -> Tensor<T, 3, TDevice, typename TDevice::template allocator_type<T>>;

#endif

// Tensor rules for allocation dynamic device memory with dimensions and optional strides provided as arrays

// Tensor(&[], &[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <Arithmetic T, size_t N, typename TDevice = device::CPU>
Tensor(const size_t(&)[N], const ssize_t(&)[N], T) -> Tensor<T, N, TDevice, typename TDevice::template allocator_type<T>>;

// Tensor(&[], &[], std::type_identity<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <Arithmetic T, size_t N, typename TDevice = device::CPU>
Tensor(const size_t(&)[N], const ssize_t(&)[N], std::type_identity<T>) -> Tensor<T, N, TDevice, typename TDevice::template allocator_type<T>>;

// Tensor(&&[], &&[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <Arithmetic T, size_t N, typename TDevice = device::CPU>
Tensor(size_t(&&)[N], ssize_t(&&)[N], T) -> Tensor<T, N, TDevice, typename TDevice::template allocator_type<T>>;

// Tensor(&&[], &&[]) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <Arithmetic T, size_t N, typename TDevice = device::CPU>
Tensor(size_t(&&)[N], ssize_t(&&)[N], std::type_identity<T>) -> Tensor<T, N, TDevice, typename TDevice::template allocator_type<T>>;

// Tensor(&[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <Arithmetic T, size_t N, typename TDevice = device::CPU>
Tensor(const size_t(&)[N], T) -> Tensor<T, N, TDevice, typename TDevice::template allocator_type<T>>;

// Tensor(&[], std::type_identity<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <Arithmetic T, size_t N, typename TDevice = device::CPU>
Tensor(const size_t(&)[N], std::type_identity<T>) -> Tensor<T, N, TDevice, typename TDevice::template allocator_type<T>>;

// Tensor(&&[], T) -> Rank-N tensor with a dynamically allocated initialized buffer.
template <Arithmetic T, size_t N, typename TDevice = device::CPU>
Tensor(const size_t(&&)[N], T) -> Tensor<T, N, TDevice, typename TDevice::template allocator_type<T>>;

// Tensor(&&[], std::type_identity<T>) -> Rank-N tensor with a dynamically allocated uninitialized buffer.
template <Arithmetic T, size_t N, typename TDevice = device::CPU>
Tensor(const size_t(&&)[N], std::type_identity<T>) -> Tensor<T, N, TDevice, typename TDevice::template allocator_type<T>>;

// Tensor(array, T)
template <Arithmetic T, size_t N, typename TDevice = device::CPU>
Tensor(std::array<size_t, N>, T) -> Tensor<T, N, TDevice, typename TDevice::template allocator_type<T>>;

// Tensor(array, array, T)
template <Arithmetic T, size_t N, typename TDevice = device::CPU>
Tensor(std::array<size_t, N>, std::array<ssize_t, N>, T) -> Tensor<T, N, TDevice, typename TDevice::template allocator_type<T>>;

// Tensor(array, std::type_identity<T>)
template <Arithmetic T, size_t N, typename TDevice = device::CPU>
Tensor(std::array<size_t, N>, std::type_identity<T>) -> Tensor<T, N, TDevice, typename TDevice::template allocator_type<T>>;

// Tensor(array, array, std::type_identity<T>)
template <Arithmetic T, size_t N, typename TDevice = device::CPU>
Tensor(std::array<size_t, N>, std::array<ssize_t, N>, std::type_identity<T>) -> Tensor<T, N, TDevice, typename TDevice::template allocator_type<T>>;

template <AnyTensor TTensor, typename TDevice = device::CPU>
Tensor(const TTensor& other)
  -> Tensor<typename TTensor::value_type, TTensor::rank, TDevice, typename TDevice::template allocator_type<typename TTensor::value_type>>;


template <libai::AnyOperator TOperator, typename TDevice = device::CPU>
Tensor(TOperator&&) -> Tensor<typename TOperator::value_type, TOperator::rank, TDevice, typename TDevice::template allocator_type<typename TOperator::value_type>>;

template <libai::AnyOperator TOperator, typename TDevice = device::CPU>
Tensor(const TOperator&) -> Tensor<typename TOperator::value_type, TOperator::rank, TDevice, typename TDevice::template allocator_type<typename TOperator::value_type>>;

/// Tensor rules for memory-mapped arguments
template <Arithmetic T, size_t N>
Tensor(const size_t(&)[N], const std::tuple<T*, size_t>&) -> Tensor<T, N, device::CPU, libai::MemoryMapped>;
template <Arithmetic T, size_t N>
Tensor(const std::array<size_t, N>&, const std::tuple<T*, size_t>&) -> Tensor<T, N, device::CPU, libai::MemoryMapped>;

// Copy/Move constructors
template <typename T, size_t NRank, typename TDevice, typename TAllocator>
Tensor(const Tensor<T, NRank, TDevice, TAllocator>&) -> Tensor<T, NRank, TDevice, TAllocator>;
template <typename T, size_t NRank, typename TDevice, typename TAllocator>
Tensor(Tensor<T, NRank, TDevice, TAllocator>&&) -> Tensor<T, NRank, TDevice, TAllocator>;

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
//
