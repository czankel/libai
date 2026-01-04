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
  using Tensor = libai::Tensor<T, R, libai::DeviceMemory<libai::device::Metal>>;

  template <typename T>
  using Array = libai::Array<T, libai::DeviceMemory<libai::device::Metal>>;
};

#else

struct TensorMetalType
{
  template <typename T, size_t TRank, typename TMemory>
  class Tensor : public libai::Tensor<T, TRank, TMemory>
  {
   public:
    using libai::Tensor<T, TRank, TMemory>::Tensor;

    Tensor(const libai::Tensor<T, TRank, libai::DeviceMemory<libai::device::Metal>>& other)
      : libai::Tensor<T, TRank, TMemory>(other) {}
    Tensor(libai::Tensor<T, TRank, libai::DeviceMemory<libai::device::Metal>>&& other)
      : libai::Tensor<T, TRank, TMemory>(other) {}
  };

  template <typename T, typename TMemory>
  class Array : public libai::Array<T, TMemory>
  {
   public:
    using libai::Array<T, TMemory>::Array;
  };
  // Tensor CTAD
  template <libai::Arithmetic T> explicit Tensor(T) -> Tensor<T, 0, libai::Scalar>;
  template <libai::Arithmetic T> explicit Tensor(std::type_identity<T>) -> Tensor<T, 0, libai::Scalar>;

  template <libai::Arithmetic... Ts>
  Tensor(Ts...) -> Tensor<std::common_type_t<Ts...>, 1, libai::StaticMemory<sizeof...(Ts)>>;

  template <libai::Arithmetic T, size_t... N>
  Tensor(T(&&... l)[N]) -> Tensor<T, 2, libai::StaticMemory<sizeof...(N), std::max({N...})>>;

  template <libai::Arithmetic T, size_t... M, size_t... N>
  Tensor(T(&&... l)[M][N]) -> Tensor<T, 3, libai::StaticMemory<sizeof...(M), std::max({M...}), std::max({N...})>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(const size_t(&)[N], const ssize_t(&)[N], T) -> Tensor<T, N, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(const size_t(&)[N], const ssize_t(&)[N], std::type_identity<T>) -> Tensor<T, N, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(size_t(&&)[N], ssize_t(&&)[N], T) -> Tensor<T, N, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(size_t(&&)[N], ssize_t(&&)[N], std::type_identity<T>) -> Tensor<T, N, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(const size_t(&)[N], T) -> Tensor<T, N, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(const size_t(&)[N], std::type_identity<T>) -> Tensor<T, N, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(const size_t(&&)[N], T) -> Tensor<T, N, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(const size_t(&&)[N], std::type_identity<T>) -> Tensor<T, N, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(std::array<size_t, N>, T) -> Tensor<T, N, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(std::array<size_t, N>, std::array<ssize_t, N>, T) -> Tensor<T, N, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(std::array<size_t, N>, std::type_identity<T>) -> Tensor<T, N, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N, typename Dev = libai::device::Metal>
  Tensor(std::array<size_t, N>, std::array<ssize_t, N>, std::type_identity<T>) -> Tensor<T, N, libai::DeviceMemory<Dev>>;

  template <libai::AnyTensor TTensor, typename Mem = libai::DeviceMemory<libai::device::Metal>>
  requires (!std::is_same_v<typename TTensor::memory_type, Mem>)
  Tensor(const TTensor& other) -> Tensor<typename TTensor::value_type, TTensor::rank, Mem>;

  template <typename TTensor, size_t TRank, typename Dev = libai::device::Metal>
  Tensor(libai::TensorView<TTensor, TRank>&&) -> Tensor<typename TTensor::value_type, TRank, libai::DeviceMemory<Dev>>;
  template <typename TTensor, size_t TRank, typename Dev = libai::device::Metal>
  Tensor(const libai::TensorView<TTensor, TRank>&) -> Tensor<typename TTensor::value_type, TRank, libai::DeviceMemory<Dev>>;

  template <libai::AnyOperator TOperator, typename Dev = libai::device::Metal>
  Tensor(TOperator&&) -> Tensor<typename TOperator::value_type, TOperator::rank, libai::DeviceMemory<Dev>>;

  template <libai::AnyOperator TOperator, typename Dev = libai::device::Metal>
  Tensor(const TOperator&) -> Tensor<typename TOperator::value_type, TOperator::rank, libai::DeviceMemory<Dev>>;

  template <libai::Arithmetic T, size_t N>
  Tensor(const size_t(&)[N], const std::tuple<T*, size_t>&) -> Tensor<T, N, libai::MemoryMapped>;
  template <libai::Arithmetic T, size_t N>
  Tensor(const std::array<size_t, N>&, const std::tuple<T*, size_t>&) -> Tensor<T, N, libai::MemoryMapped>;

  template <typename T, size_t R, typename M>
  Tensor(const libai::Tensor<T, R, M>&) -> Tensor<T, R, libai::DeviceMemory<libai::device::Metal>>;
  template <typename T, size_t R, typename M>
  Tensor(libai::Tensor<T, R, M>&&) -> Tensor<T, R, libai::DeviceMemory<libai::device::Metal>>;

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
