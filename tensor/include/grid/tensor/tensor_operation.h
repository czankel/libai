//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_TENSOR_OPERATION_H
#define GRID_TENSOR_TENSOR_OPERATION_H

namespace libai {


/// TensorOperation is a base class and wrapper for tensor operations.
///
/// Aliasing with partial specialization requires any CTAD rule to have a template-parameter-list
/// on the rhs that matches the aliasing definition This prohibites the use of more "fancy" rules,
/// such as defining the rank or type of a tensor.
///
/// The following rule, for example, would be ignored when aliasing with partial specialization
/// is used:
///
///   SomeClass(_args_) -> SomeClass<FancyType<_args_>>;
///
/// The TensorOperation exposes the necessary template parameters (value type and rank) to deduce
/// the same parameters in the Tensor class. This allowes derived operator classes to be used as
/// arguments in the Tensor constructor or assign operator.
///
/// @tparam T          value type
/// @tparam TRank      rank
/// @tparam TOperation wrapped operation

template <typename T, size_t TRank, typename TOperation>
class TensorOperation
{
 public:
  using value_type = T;
  constexpr static size_t rank = TRank;

  /// Constructo
  /// @param op   Tensor operator object
  TensorOperation(TOperation& op) : op_(op) {}

  /// operator() executes the operation.
  auto operator()() const                             { return op_(); }

 private:
  TOperation& op_;
};

} // end of namespace libai

#endif	// GRID_TENSOR_TENSOR_OPERATION_H
