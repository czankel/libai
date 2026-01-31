//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// Important: Class template argument deduction for alias templates P1814R0 not supported on all
// compilers. This requires to duplicate *all* deduction rules in slowcpu/tensor.h

#include <memory>

#include <libai/tensor/concepts.h>

namespace libai {

template <typename T, size_t NRank, typename TDevice, typename TAllocator> class Tensor;

} // end of namespace libai

#if (__cpp_deduction_guides >= 201907L || \
    (__APPLE__ && (__clang_major__ > 17 || (__clang_major__ == 17 && __clang_minor__ >= 0))))

struct TensorCPUType
{
  template <typename T, size_t N>
  using Tensor = libai::Tensor<T, N, libai::device::CPU, std::allocator<T>>;

  template <typename T>
  using Array = libai::Array<T, std::allocator<T>>;
};

#else

struct TensorCPUType
{
  template <typename T, size_t NRank, typename TDevice, typename TAllocator = std::allocator<T>>
  class Tensor : public libai::Tensor<T, NRank, TDevice, TAllocator>
  {
   public:
    using libai::Tensor<T, NRank, TDevice, TAllocator>::Tensor;

    template <typename A>
    Tensor(const libai::Tensor<T, NRank, TDevice, A>& other)
      : libai::Tensor<T, NRank, TDevice, TAllocator>(other) {}

    template <typename A>
    Tensor(libai::Tensor<T, NRank, TDevice, A>&& other)
      : libai::Tensor<T, NRank, TDevice, TAllocator>(other) {}
  };

  template <typename T, typename TAllocator>
  class Array : public libai::Array<T, TAllocator>
  {
   public:
    using libai::Array<T, TAllocator>::Array;
  };

  // Tensor CTAD

#if 0  // Scalar
  template <libai::Arithmetic T> explicit Tensor(T) -> Tensor<T, 0, libai::device::CPU, libai::Scalar>;
  template <libai::Arithmetic T> explicit Tensor(std::type_identity<T>) -> Tensor<T, 0, libai::device::CPU, libai::Scalar>;
#endif

  // constructor from initializer list

  template <libai::Arithmetic... Ts>
  Tensor(Ts...)
    -> Tensor<std::common_type_t<Ts...>, 1,
              libai::device::CPU, std::allocator<std::common_type_t<Ts...>>>;

  template <libai::Arithmetic T, size_t... N>
  Tensor(T(&&... l)[N]) -> Tensor<T, 2, libai::device::CPU, std::allocator<T>>;

  template <libai::Arithmetic T, size_t... M, size_t... N>
  Tensor(T(&&... l)[M][N]) -> Tensor<T, 3, libai::device::CPU, std::allocator<T>>;

  // constructor from dimensions, strides, with and without initializer

  template <libai::Arithmetic T, size_t N>
  Tensor(const size_t(&)[N], const ssize_t(&)[N], T)
    -> Tensor<T, N, libai::device::CPU, std::allocator<T>>;

  template <libai::Arithmetic T, size_t N>
  Tensor(const size_t(&)[N], const ssize_t(&)[N], std::type_identity<T>)
    -> Tensor<T, N, libai::device::CPU, std::allocator<T>>;

  template <libai::Arithmetic T, size_t N>
  Tensor(size_t(&&)[N], ssize_t(&&)[N], T) -> Tensor<T, N, libai::device::CPU, std::allocator<T>>;

  template <libai::Arithmetic T, size_t N>
  Tensor(size_t(&&)[N], ssize_t(&&)[N], std::type_identity<T>) -> Tensor<T, N, libai::device::CPU, std::allocator<T>>;

  template <libai::Arithmetic T, size_t N>
  Tensor(const size_t(&)[N], T) -> Tensor<T, N, libai::device::CPU, std::allocator<T>>;

  template <libai::Arithmetic T, size_t N>
  Tensor(const size_t(&)[N], std::type_identity<T>) -> Tensor<T, N, libai::device::CPU, std::allocator<T>>;

  template <libai::Arithmetic T, size_t N>
  Tensor(const size_t(&&)[N], T) -> Tensor<T, N, libai::device::CPU, std::allocator<T>>;

  template <libai::Arithmetic T, size_t N>
  Tensor(const size_t(&&)[N], std::type_identity<T>) -> Tensor<T, N, libai::device::CPU, std::allocator<T>>;

  template <libai::Arithmetic T, size_t N>
  Tensor(std::array<size_t, N>, T) -> Tensor<T, N, libai::device::CPU, std::allocator<T>>;

  template <libai::Arithmetic T, size_t N>
  Tensor(std::array<size_t, N>, std::array<ssize_t, N>, T)
    -> Tensor<T, N, libai::device::CPU, std::allocator<T>>;

  template <libai::Arithmetic T, size_t N>
  Tensor(std::array<size_t, N>, std::type_identity<T>)
    -> Tensor<T, N, libai::device::CPU, std::allocator<T>>;

  template <libai::Arithmetic T, size_t N>
  Tensor(std::array<size_t, N>, std::array<ssize_t, N>, std::type_identity<T>)
    -> Tensor<T, N, libai::device::CPU, std::allocator<T>>;

#if 0 // constructor from other tensor
  template <libai::AnyTensor TTensor>
  requires (!std::is_same_v<typename TTensor::device_type, libai::device::CPU> ||
            !std::is_same_v<typename TTensor::allocator_type, std::allocator<typename TTensor::value_type>>)
  Tensor(const TTensor& other)
    -> Tensor<typename TTensor::value_type, TTensor::rank,
              libai::device::CPU, std::allocator<typename TTensor::value_type>>;
#endif

  // copy/move constructor

  template <typename T, size_t R, typename D, typename A>
  Tensor(const Tensor<T, R, D, A>&) -> Tensor<T, R, D, A>;

  template <typename T, size_t R, typename D, typename A>
  Tensor(Tensor<T, R, D, A>&&) -> Tensor<T, R, D, A>;

  // constructor from TensorView

  template <typename TTensor, size_t NRank>
  Tensor(libai::TensorView<TTensor, NRank>&&)
    -> Tensor<typename TTensor::value_type, NRank,
              libai::device::CPU, std::allocator<typename TTensor::value_type>>;

  template <typename TTensor, size_t NRank>
  Tensor(const libai::TensorView<TTensor, NRank>&)
    -> Tensor<typename TTensor::value_type, NRank,
              libai::device::CPU, std::allocator<typename TTensor::value_type>>;

  // constructor from Operator

  template <libai::AnyOperator TOperator>
  Tensor(TOperator&&)
    -> Tensor<typename TOperator::value_type, TOperator::rank,
              libai::device::CPU, std::allocator<typename TOperator::value_type>>;

  template <libai::AnyOperator TOperator>
  Tensor(const TOperator&)
    -> Tensor<typename TOperator::value_type, TOperator::rank,
              libai::device::CPU, std::allocator<typename TOperator::value_type>>;

  // CTAD for Array

  template <typename T>
  Array(size_t, T) -> Array<T, std::allocator<T>>;

  template <typename T>
  Array(size_t, std::type_identity<T>) -> Array<T, std::allocator<T>>;

  template <typename T, size_t N>
  Array(const std::array<size_t, N>&, const std::array<ssize_t, N>&, T) -> Array<T, std::allocator<T>>;

  template <typename T, size_t N>
  Array(const std::array<size_t, N>&, const std::array<ssize_t, N>&, std::type_identity<T>)
    -> Array<T, std::allocator<T>>;
};

#endif
