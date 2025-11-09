//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_FUNCTION_H
#define GRID_TENSOR_FUNCTION_H

#include <algorithm>
#include <ranges>
#include <span>
#include <tuple>

#include "concepts.h"
#include "tensor_operation.h"

namespace libai {

/// @brief Function is a wrapper for a device-specific function operator implementation.
///
/// Function provides a lazy-implementation that only stores the tensor and any additional
/// parameters, and evaluates the function with operator().
///
/// Function is typically not used directly, instead, use the actual functions,
/// such as RmsNorm() or Rope().
///
/// The actual device implementations need to provide an operator() with an input and output
/// range. This differs from, e.g. std::ranges::transform that requires an output iterator
/// instead of a range.
///
///  template<std::ranges::input_range, std::ranges::output_range> operator();
///
///  @tparm TOperator unary operator type
///  @tparm TTensor  tensor type
///
template <typename TOperator, AnyTensor TTensor, typename... Args>
class Function : public TensorOperation<typename std::remove_cvref_t<TTensor>::value_type,
                                        std::remove_cvref_t<TTensor>::rank,
                                        Function<TOperator, TTensor, Args...>>
{
 public:
  using typename Function::TensorOperation::value_type;
  using Function::TensorOperation::rank;

  template <typename T>
  Function(TOperator, T&& tensor, Args&&... args)
    : TensorOperation<value_type, rank, Function<TOperator, TTensor, Args...>>(*this),
      tensor_(std::forward<T>(tensor)),
      args_(std::forward<Args>(args)...)
  { }

  ~Function() {}

  Function() = delete;
  Function(const Function& other) = delete;
  Function& operator=(const Function& other) = delete;

 public:

  /// operator()() evaluates the rope operator and returns a tensor.
  auto operator()() const
  {
    using ResultTensor = Tensor<value_type, rank, DeviceMemory<tensor_device_t<TTensor>>>;
    auto result = ResultTensor(tensor_.Dimensions(), Uninitialized<value_type>{});
    std::apply(operator_, std::tuple_cat(std::forward_as_tuple(tensor_, result), args_));
    return result;
  }

  /// Rank returns the rank of the tensor.
  size_t Rank() const                                     { return rank; }

  /// Dimensions returns the dimensions for the axis.
  const std::array<size_t, rank>& Dimensions() const      { return tensor_.dimensions_; }

  /// Strides returns the strides for the axis.
  const std::array<ssize_t, rank>& Strides() const        { return tensor_.strides_; }

  /// Size returns the data buffer size.
  size_t Size() const                                     { return tensor_.size_; }


 private:
  static TOperator operator_;
  TTensor tensor_;
  std::tuple<std::remove_reference_t<Args>...> args_;
};

template <typename TOp, typename T, typename... Args> Function(TOp, T&&, Args&&...)
  -> Function<TOp, typename to_tensor<T>::type, Args...>;
template <typename TOperator, AnyTensor TTensor, typename... Args>
TOperator Function<TOperator, TTensor, Args...>::operator_;


template <typename> class RmsNormOperator;
template <typename> class RopeOperator;
template <typename> class SoftMaxOperator;


/// @brief RmsNorm returns a tensor of the RMS normalized tensor.
template <TensorConvertible TTensor>
requires (std::remove_cvref_t<TTensor>::rank <= 2)
auto RmsNorm(TTensor&& tensor)
{
  return Function(RmsNormOperator<tensor_device_t<TTensor>>(), std::forward<TTensor>(tensor));
}

/// @brief Rope returns a tensor with RoPE calculations for position Pos, applied to the provided tensor.
template <TensorConvertible TTensor>
auto Rope(TTensor&& tensor, int pos)
{
  return Function(RopeOperator<tensor_device_t<TTensor>>(), std::forward<TTensor>(tensor), pos);
}

/// @brief SoftMax returns a tensor with the SoftMax applied to the provided tensor.
template <TensorConvertible TTensor>
auto SoftMax(TTensor&& tensor)
{
  return Function(SoftMaxOperator<tensor_device_t<TTensor>>(), std::forward<TTensor>(tensor));
}

} // end of namespace libai

#endif  // GRID_TENSOR_FUNCTION_H
