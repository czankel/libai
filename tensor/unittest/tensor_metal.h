//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// Important: Class template argument deduction for alias templates P1814R0 not supported on all
// compilers. This requires to duplicate *all* deduction rules in slowcpu/tensor.h


#if __cpp_deduction_guides >= 201907L

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
  };

  // dynamic tensors
  template <typename T, size_t N>
  Tensor(const size_t(&)[N], const ssize_t(&)[N], T) -> Tensor<T, N, libai::DeviceMemory<libai::device::Metal>>;
  template <typename T, size_t N>
  Tensor(const size_t(&)[N], const ssize_t(&)[N], libai::Uninitialized<T>) -> Tensor<T, N, libai::DeviceMemory<libai::device::Metal>>;
  template <typename T, size_t N>
  Tensor(size_t(&&)[N], ssize_t(&&)[N], T) -> Tensor<T, N, libai::DeviceMemory<libai::device::Metal>>;
  template <typename T, size_t N>
  Tensor(size_t(&&)[N], ssize_t(&&)[N], libai::Uninitialized<T>) -> Tensor<T, N, libai::DeviceMemory<libai::device::Metal>>;
  template <typename T, size_t N>
  Tensor(const size_t(&)[N], T) -> Tensor<T, N, libai::DeviceMemory<libai::device::Metal>>;
  template <typename T, size_t N>
  Tensor(const size_t(&)[N], libai::Uninitialized<T>) -> Tensor<T, N, libai::DeviceMemory<libai::device::Metal>>;
  template <typename T, size_t N>
  Tensor(const size_t(&&)[N], T) -> Tensor<T, N, libai::DeviceMemory<libai::device::Metal>>;
  template <typename T, size_t N>
  Tensor(const size_t(&&)[N], libai::Uninitialized<T>) -> Tensor<T, N, libai::DeviceMemory<libai::device::Metal>>;
  template <typename T, size_t N>
  Tensor(std::array<size_t, N>, T) -> Tensor<T, N, libai::DeviceMemory<libai::device::Metal>>;
  template <typename T, size_t N>
  Tensor(std::array<size_t, N>, std::array<ssize_t, N>, T) -> Tensor<T, N, libai::DeviceMemory<libai::device::Metal>>;
  template <typename T, size_t N>
  Tensor(std::array<size_t, N>, libai::Uninitialized<T>) -> Tensor<T, N, libai::DeviceMemory<libai::device::Metal>>;
  template <typename T, size_t N>
  Tensor(std::array<size_t, N>, std::array<ssize_t, N>, libai::Uninitialized<T>) -> Tensor<T, N, libai::DeviceMemory<libai::device::Metal>>;

  // memory-mapped tensors
  template <libai::Arithmetic T, size_t N>
  Tensor(const size_t(&)[N], const std::tuple<T*, size_t>&) -> Tensor<T, N, libai::MemoryMapped>;
  template <libai::Arithmetic T, size_t N>
  Tensor(const std::array<size_t, N>&, const std::tuple<T*, size_t>&) -> Tensor<T, N, libai::MemoryMapped>;

  // copy & move constructors
  template <typename T, size_t N, typename M>
  Tensor(const libai::Tensor<T, N, M>&) -> Tensor<T, N, libai::DeviceMemory<libai::device::Metal>>;
  template <typename T, size_t N, typename M>
  Tensor(libai::Tensor<T, N, M>&&) -> Tensor<T, N, libai::DeviceMemory<libai::device::Metal>>;

  // tensor view
  template <template <typename, size_t> typename TensorView, typename TTensor, size_t TRank>
  Tensor(TensorView<TTensor, TRank>&&) -> Tensor<typename TTensor::value_type, TRank, libai::DeviceMemory<libai::device::Metal>>;
  template <template <typename, size_t> typename TensorView, typename TTensor, size_t TRank>
  Tensor(const TensorView<TTensor, TRank>&) -> Tensor<typename TTensor::value_type, TRank, libai::DeviceMemory<libai::device::Metal>>;

  // operators
  template <libai::AnyOperator TOperator>
  Tensor(const TOperator&) ->
    Tensor<typename TOperator::value_type, TOperator::rank, libai::DeviceMemory<libai::device::Metal>>;

  // Array
  template <typename T>
  Array(size_t, T) -> Array<T, libai::DeviceMemory<libai::device::Metal>>;

  template <typename T, typename Mem = libai::DeviceMemory<device::CPU>>
  Array(size_t, Uninitialized<T>) -> Array<T, libai::DeviceMemory<libai::device::Metal>>;

  template <typename T, size_t N>
  Array(const std::array<size_t, N>&, const std::array<ssize_t, N>&, T)
    -> Array<T, libai::DeviceMemory<libai::device::Metal>>;

  template <typename T, size_t N>
  Array(const std::array<size_t, N>&, const std::array<ssize_t, N>&, Uninitialized<T>)
    -> Array<T, libai::DeviceMemory<libai::device::Metal>>;

};

#endif
