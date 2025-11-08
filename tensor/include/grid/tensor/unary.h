//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef LIBAI_TENSOR_UNARY_H
#define LIBAI_TENSOR_UNARY_H

#include <algorithm>
#include <ranges>
#include <span>

#include "concepts.h"
#include "tensor_operation.h"

namespace libai {

template <template <typename> typename, typename> class UnaryOperation;

/// @brief Unary is a wrapper for a device-specific unary operations.
///
/// Unary provides a lazy-implementation that only stores the tensor and evaluates
/// the operation with operator().
///
/// Unary is typically not used directly, instead, use the actual functions, such as Neg(Tensor).
///
/// The actual implementation needs to provide an operator() with an input and output range.
/// This differs from, e.g. std::ranges::transform that requires an output iterator instead
/// of a range.
///
///  template<std::ranges::input_range, std::ranges::output_range> operator();
///
///  @tparm TOperation unary operator type
///  @tparm TTensor  tensor type
///
template <typename TOperation, AnyTensor TTensor>
class Unary : public TensorOperation<typename std::remove_cvref_t<TTensor>::value_type,
                                     std::remove_cvref_t<TTensor>::rank,
                                     Unary<TOperation, TTensor>>
{
 public:
  using typename Unary::TensorOperation::value_type;
  using Unary::TensorOperation::rank;

  template <typename T>
  Unary(TOperation, T&& tensor)
    : TensorOperation<value_type, rank, Unary<TOperation, TTensor>>(*this),
      tensor_(std::forward<T>(tensor))
  {}

  ~Unary() {}

  Unary() = delete;
  Unary(const Unary& other) = delete;
  Unary& operator=(const Unary& other) = delete;

 public:

  /// operator()() evaluates the unary operator and returns a tensor.
  auto operator()() const
  {
    using ResultTensor = Tensor<value_type, rank, DeviceMemory<tensor_device_t<TTensor>>>;
    auto result = ResultTensor(tensor_.Dimensions(), Uninitialized<value_type>{});
    operator_(tensor_, result);
    return result;
  }

 private:
  static TOperation operator_;
  TTensor tensor_;
};

template <typename TOp, typename T> Unary(TOp, T&&) -> Unary<TOp, typename to_tensor<T>::type>;

template <typename TOperation, AnyTensor TTensor> TOperation Unary<TOperation, TTensor>::operator_;

//
// Elementary Unary Operators
//

template <typename> struct CopyOperator;
template <typename> struct NegOperator;

/// @brief Copy returns a copy of the tensor.
template <TensorConvertible TTensor>
auto Copy(TTensor&& tensor)
{
  return Unary(UnaryOperation<CopyOperator, tensor_device_t<TTensor>>(), std::forward<TTensor>(tensor));
}

/// @brief Neg returns a copy of the negated tensor.
template <TensorConvertible TTensor>
auto Neg(TTensor&& tensor)
{
  return Unary(UnaryOperation<NegOperator, tensor_device_t<TTensor>>(), std::forward<TTensor>(tensor));
}

//
// Unary Operations
//

template <typename> struct SiluFunction;

/// @brief Silu returns a tensor with SiLU activation applied to the provided tensor.
template <TensorConvertible TTensor>
auto Silu(TTensor&& tensor)
{
  return Unary(UnaryOperation<SiluFunction, tensor_device_t<TTensor>>(), std::forward<TTensor>(tensor));
}


} // end of namespace grd

#endif  // LIBAI_TENSOR_UNARY_H
