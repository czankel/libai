//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// Important: Class template argument deduction for alias templates P1814R0 not supported on all
// compilers. This requires to duplicate *all* deduction rules in slowcpu/tensor.h


#if (__cpp_deduction_guides >= 201907L || \
    (__APPLE__ && (__clang_major__ > 17 || (__clang_major__ == 17 && __clang_minor__ >= 0))))
struct TensorMetalType
{
  template <typename T, size_t R>
  using Tensor = libai::Tensor<T, R, libai::device::Metal, libai::DeviceMemory<libai::device::Metal>>;

  template <typename T>
  using Array = libai::Array<T, libai::DeviceMemory<libai::device::Metal>>;
};

#else

struct TensorMetalType
{
  template <typename T, size_t TRank, typename TDevice, typename TMemory>
  class Tensor : public libai::Tensor<T, TRank, TDevice, TMemory>
  {
   public:
    using libai::Tensor<T, TRank, TDevice, TMemory>::Tensor;
    using memory_type = libai::DeviceMemory<libai::device::Metal>;

    Tensor(const libai::Tensor<T, TRank, libai::device::Metal, memory_type>& other)
      : libai::Tensor<T, TRank, TDevice, TMemory>(other) {}
    Tensor(libai::Tensor<T, TRank, libai::device::Metal, memory_type>&& other)
      : libai::Tensor<T, TRank, TDevice, TMemory>(other) {}
  };

  template <typename T, typename TMemory>
  class Array : public libai::Array<T, TMemory>
  {
   public:
    using libai::Array<T, TMemory>::Array;
  };

  // Tensor CTAD
  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(const size_t(&)[N], const ssize_t(&)[N], T) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(const size_t(&)[N], const ssize_t(&)[N], std::type_identity<T>) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(size_t(&&)[N], ssize_t(&&)[N], T) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(size_t(&&)[N], ssize_t(&&)[N], std::type_identity<T>) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(const size_t(&)[N], T) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(const size_t(&)[N], std::type_identity<T>) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(const size_t(&&)[N], T) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(const size_t(&&)[N], std::type_identity<T>) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(std::array<size_t, N>, T) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(std::array<size_t, N>, std::array<ssize_t, N>, T) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(std::array<size_t, N>, std::type_identity<T>) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(std::array<size_t, N>, std::array<ssize_t, N>, std::type_identity<T>) -> Tensor<T, N, Dev, libai::DeviceMemory<Dev>>;

  template <libai::AnyTensor TTensor, typename Mem = libai::DeviceMemory<libai::device::Metal>>
  requires (!std::is_same_v<typename TTensor::memory_type, Mem>)
  Tensor(const TTensor& other) -> Tensor<typename TTensor::value_type, TTensor::rank, libai::device::Metal, Mem>;

  template <typename TTensor, size_t TRank, typename Dev = libai::device::Metal>
  Tensor(libai::TensorView<TTensor, TRank>&&) -> Tensor<typename TTensor::value_type, TRank, Dev, libai::DeviceMemory<Dev>>;
  template <typename TTensor, size_t TRank, typename Dev = libai::device::Metal>
  Tensor(const libai::TensorView<TTensor, TRank>&) -> Tensor<typename TTensor::value_type, TRank, Dev, libai::DeviceMemory<Dev>>;

  template <libai::AnyOperator TOperator, typename Dev = libai::device::Metal>
  Tensor(TOperator&&) -> Tensor<typename TOperator::value_type, TOperator::rank, Dev, libai::DeviceMemory<Dev>>;

  template <libai::AnyOperator TOperator, typename Dev = libai::device::Metal>
  Tensor(const TOperator&) -> Tensor<typename TOperator::value_type, TOperator::rank, Dev, libai::DeviceMemory<Dev>>;

  // CTAD for Array
  template <typename T>
  Array(size_t, T) -> Array<T, libai::DeviceMemory<libai::device::Metal>>;

  template <typename T, typename Mem = libai::DeviceMemory<libai::device::Metal>>
  Array(size_t, std::type_identity<T>) -> Array<T, libai::DeviceMemory<libai::device::Metal>>;

  template <typename T, size_t N>
  Array(const std::array<size_t, N>&, const std::array<ssize_t, N>&, T)
    -> Array<T, libai::DeviceMemory<libai::device::Metal>>;

  template <typename T, size_t N>
  Array(const std::array<size_t, N>&, const std::array<ssize_t, N>&, std::type_identity<T>)
    -> Array<T, libai::DeviceMemory<libai::device::Metal>>;
};

#endif
