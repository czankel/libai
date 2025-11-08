//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef LIBAI_TENSOR_MATMUL_H
#define LIBAI_TENSOR_MATMUL_H

#include <span>
#include <algorithm>
#include <ranges>

#include "concepts.h"

namespace libai {

template <typename> class MatmulOperator;
template <typename, size_t, typename> class Tensor;

namespace
{
  template <typename TTensor1, typename TTensor2>
  struct MatmulRank
  {
    using tensor1_type = std::remove_reference_t<TTensor1>;
    using tensor2_type = std::remove_reference_t<TTensor2>;
    constexpr static size_t rank =
      tensor1_type::rank != 1 || tensor2_type::rank != 1 ? std::min(tensor1_type::rank, tensor2_type::rank) : 0;
  };
}

// @brief MatMul is a wrapper for a device-specific matmul operator implementation
//
// Matmul provides a lazy-implementation that only stores the tensors and evaluates
// the operation with operator().
//
// Matmul only supports matrix multiplications of rank-2 matrices, matrix and vector, and
// vector dot.
//
// TODO: add support for higher-rank tensors?
template <TensorConvertible TTensor1, TensorConvertible TTensor2>
class Matmul : TensorOperation<std::common_type_t<typename std::remove_cvref_t<TTensor1>::value_type,
                                                  typename std::remove_cvref_t<TTensor2>::value_type>,
                              MatmulRank<TTensor1, TTensor2>::rank,
                              Matmul<TTensor1, TTensor2>>
{
  using device = tensor_device_t<TTensor1>;

 public:
  using Matmul::TensorOperation::rank;
  using tensor1_type = std::remove_reference_t<TTensor1>;
  using tensor2_type = std::remove_reference_t<TTensor2>;
  using value_type = std::common_type_t<typename tensor1_type::value_type, typename tensor2_type::value_type>;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  constexpr static size_t tensor1_rank = tensor1_type::rank;
  constexpr static size_t tensor2_rank = tensor2_type::rank;

  template <typename T1, typename T2>
  requires (tensor1_rank > 0 && tensor2_rank > 0)
  Matmul(T1&& tensor1, T2&& tensor2)
   : TensorOperation<value_type, rank, Matmul<TTensor1, TTensor2>>(*this),
     tensor1_(std::forward<T1>(tensor1)),
     tensor2_(std::forward<T2>(tensor2))
  {
    if constexpr (tensor1_rank > 0 && tensor2_rank > 0)
    {
      // matmul: lhs columns (dim[rank-1])  and rhs rows (dim[rank-2]) have to match; assume column-vector
      size_t dim = tensor2_rank > 1 ? tensor2_rank - 2 : 0;
      size_t lhs = tensor1_.Dimensions()[tensor1_rank - 1];
      size_t rhs = tensor2_.Dimensions()[dim];
      if (lhs != rhs)
        throw std::runtime_error("dimensions don't match: lhs: " + std::to_string(lhs) + " rhs: " + std::to_string(rhs));
    }
  }

  // delete assignment and copy/move constructors
  Matmul() = delete;
  Matmul(const Matmul& other) = delete;
  Matmul& operator=(const Matmul& other) = delete;

  /// operator()() executes and returns a (scalar) tensor with the 'vector dot' multiplication.
  auto operator()() const requires (tensor1_rank == 1 && tensor2_rank == 1)
  {
    if (tensor1_.Dimensions()[0] != tensor2_.Dimensions()[0])
      throw std::runtime_error("mismatching dimensions in vector product");

    auto result = Tensor<value_type, 0, DeviceMemory<device>>{value_type{0}};
    operator_(tensor1_, tensor2_, result);
    return result;
  }

  /// operator()() executes and returns a (matrix) tensor for a mtrix multiplication.
  auto operator()() const requires (tensor1_rank == 2 && tensor2_rank == 2)
  {
    auto&& dims1 = tensor1_.Dimensions();
    auto&& dims2 = tensor2_.Dimensions();
    if (dims1[1] != dims2[0])
      throw std::runtime_error("mismatching dimensions in matrix multiplication");

    auto result = Tensor<value_type, 2, DeviceMemory<device>>({dims1[0], dims2[1]}, Uninitialized<value_type>{});
    operator_(tensor1_, tensor2_, result);
    return result;
  }

  /// operator()() executes and returns a (vector) tensor of a matrix * vector multiplication.
  auto operator()() const requires (tensor1_rank == 2 && tensor2_rank == 1)
  {
    auto&& dims1 = tensor1_.Dimensions();
    auto&& dims2 = tensor2_.Dimensions();
    if (dims1[1] != dims2[0])
      throw std::runtime_error("mismatching dimensions in matrix multiplication");

    auto result = Tensor<value_type, 1, DeviceMemory<device>>(dims1[0], Uninitialized<value_type>{});
    operator_(tensor1_, tensor2_, result);
    return result;
  }

  /// operator()() executes and returns a (vector) tensor of a vector * matrix multiplication.
  auto operator()() const requires (tensor1_rank == 1 && tensor2_rank == 2)
  {
    auto&& dims1 = tensor1_.Dimensions();
    auto&& dims2 = tensor2_.Dimensions();
    if (dims1[0] != dims2[0])
      throw std::runtime_error("mismatching dimensions in matrix multiplication");

    auto result = Tensor<value_type, 1, DeviceMemory<device>>(dims2[1], Uninitialized<value_type>{});
    operator_(tensor1_, tensor2_, result);
    return result;
  }

 private:
  MatmulOperator<device> operator_;
  TTensor1 tensor1_;
  TTensor2 tensor2_;
};

template <typename T1, typename T2> Matmul(T1&&, T2&&)
  -> Matmul<typename to_tensor<T1>::type, typename to_tensor<T2>::type>;

} // end of namespace grd

#endif  // LIBAI_TENSOR_MATMUL_H
