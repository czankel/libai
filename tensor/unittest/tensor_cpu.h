//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// Important: Class template argument deduction for alias templates P1814R0 not supported on all
// compilers. This requires to duplicate *all* deduction rules in slowcpu/tensor.h

#include <libai/tensor/concepts.h>

namespace libai {

template <typename, size_t, typename, typename> class Tensor;

} // end of namespace libai

#if (__cpp_deduction_guides >= 201907L || \
    (__APPLE__ && (__clang_major__ > 17 || (__clang_major__ == 17 && __clang_minor__ >= 0))))
struct TensorCPUType
{
  template <typename T, size_t N>
  using Tensor = libai::Tensor<T, N, libai::device::CPU, libai::DeviceMemory<libai::device::CPU>>;

  template <typename T>
  using Array = libai::Array<T, libai::DeviceMemory<libai::device::CPU>>;
};

#else

struct TensorCPUType
{
  template <typename T, size_t NRank, typename TDevice, typename TMemory>
  class Tensor : public libai::Tensor<T, NRank, TDevice, TMemory>
  {
   public:
    using libai::Tensor<T, NRank, TDevice, TMemory>::Tensor;

    Tensor(const libai::Tensor<T, NRank, TDevice, TMemory>& other) : libai::Tensor<T, NRank, TDevice, TMemory>(other) {}
    Tensor(libai::Tensor<T, NRank, TDevice, TMemory>&& other) : libai::Tensor<T, NRank, TDevice, TMemory>(std::move(other)) {}
  };

  template <typename T, typename TMemory>
  class Array : public libai::Array<T, TMemory>
  {
   public:
    using libai::Array<T, TMemory>::Array;
  };

  // Tensor CTAD
  template <libai::Arithmetic T> explicit Tensor(T) -> Tensor<T, 0, libai::device::CPU, libai::Scalar>;
  template <libai::Arithmetic T> explicit Tensor(std::type_identity<T>) -> Tensor<T, 0, libai::device::CPU, libai::Scalar>;

  template <libai::Arithmetic... Ts>
  Tensor(Ts...) -> Tensor<std::common_type_t<Ts...>, 1, libai::device::CPU, libai::StaticResource<sizeof...(Ts)>>;

  template <libai::Arithmetic T, size_t... N>
  Tensor(T(&&... l)[N]) -> Tensor<T, 2, libai::device::CPU, libai::StaticResource<sizeof...(N), std::max({N...})>>;

  template <libai::Arithmetic T, size_t... M, size_t... N>
  Tensor(T(&&... l)[M][N]) -> Tensor<T, 3, libai::device::CPU, libai::StaticResource<sizeof...(M), std::max({M...}), std::max({N...})>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::CPU>
  Tensor(const size_t(&)[N], const ssize_t(&)[N], T) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::CPU>
  Tensor(const size_t(&)[N], const ssize_t(&)[N], std::type_identity<T>) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::CPU>
  Tensor(size_t(&&)[N], ssize_t(&&)[N], T) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::CPU>
  Tensor(size_t(&&)[N], ssize_t(&&)[N], std::type_identity<T>) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::CPU>
  Tensor(const size_t(&)[N], T) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::CPU>
  Tensor(const size_t(&)[N], std::type_identity<T>) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::CPU>
  Tensor(const size_t(&&)[N], T) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::CPU>
  Tensor(const size_t(&&)[N], std::type_identity<T>) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::CPU>
  Tensor(std::array<size_t, N>, T) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::CPU>
  Tensor(std::array<size_t, N>, std::array<ssize_t, N>, T) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::CPU>
  Tensor(std::array<size_t, N>, std::type_identity<T>) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::CPU>
  Tensor(std::array<size_t, N>, std::array<ssize_t, N>, std::type_identity<T>) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::AnyTensor TTensor, typename Dev = libai::device::CPU, typename Alloc = libai::DeviceMemory<libai::device::CPU>>
  requires (!std::is_same_v<typename TTensor::device_type, Dev> ||
            !std::is_same_v<typename TTensor::allocator_type, Alloc>)
  Tensor(const TTensor& other) -> Tensor<typename TTensor::value_type, TTensor::rank, Dev, Alloc>;

  template <typename TTensor, size_t NRank, typename Dev = libai::device::CPU>
  Tensor(libai::TensorView<TTensor, NRank>&&) -> Tensor<typename TTensor::value_type, NRank, Dev, libai::DeviceMemory<Dev>>;
  template <typename TTensor, size_t NRank, typename Dev = libai::device::CPU>
  Tensor(const libai::TensorView<TTensor, NRank>&) -> Tensor<typename TTensor::value_type, NRank, Dev, libai::DeviceMemory<Dev>>;

  template <libai::AnyOperator TOperator, typename Dev = libai::device::CPU>
  Tensor(TOperator&&) -> Tensor<typename TOperator::value_type, TOperator::rank, Dev, libai::DeviceMemory<Dev>>;

  template <libai::AnyOperator TOperator, typename Dev = libai::device::CPU>
  Tensor(const TOperator&) -> Tensor<typename TOperator::value_type, TOperator::rank, Dev, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N>
  Tensor(const size_t(&)[N], const std::tuple<T*, size_t>&) -> Tensor<T, N, libai::device::CPU, libai::MemoryMapped>;
  template <libai::Arithmetic T, size_t N>
  Tensor(const std::array<size_t, N>&, const std::tuple<T*, size_t>&) -> Tensor<T, N, libai::device::CPU, libai::MemoryMapped>;

  // CTAD for Array
  template <typename T>
  Array(size_t, T) -> Array<T, libai::DeviceMemory<libai::device::CPU>>;

  template <typename T, typename Mem = libai::DeviceMemory<libai::device::CPU>>
  Array(size_t, std::type_identity<T>) -> Array<T, libai::DeviceMemory<libai::device::CPU>>;

  template <typename T, size_t N>
  Array(const std::array<size_t, N>&, const std::array<ssize_t, N>&, T)
    -> Array<T, libai::DeviceMemory<libai::device::CPU>>;

  template <typename T, size_t N>
  Array(const std::array<size_t, N>&, const std::array<ssize_t, N>&, std::type_identity<T>)
    -> Array<T, libai::DeviceMemory<libai::device::CPU>>;
};

#endif
