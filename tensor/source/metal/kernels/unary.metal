//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <metal_math>

#include "../../instantiate.h"

#include "utils.h"

struct CopyOperator { template<typename T> inline T operator()(T x) { return x; } };
struct NegOperator  { template<typename T> inline T operator()(T x) { return -x; } };
struct SiluOperator { template<typename T> inline T operator()(T x)
 { return static_cast<T>(x / (T{1} + metal::precise::exp(-x))); } };

//
// Fast unary opeator supporting scalars but without strides
//

template <typename Op, typename T, typename U>
[[kernel]] void UnaryOperationS(device U* d,
                               device const T* x,
                               uint index [[thread_position_in_grid]])
{
  d[index] = Op()(x[index]);
}

template <typename Op, typename T, typename U>
[[kernel]] void UnaryOperationV(device U* d,
                               device const T* x,
                               uint index [[thread_position_in_grid]])
{
  d[index] = Op()(x[index]);
}


#define FAST_FUNCTION(R, O, T) \
  template [[host_name(stringify(UnaryOperation ## R ## O ## T))]]  \
  [[kernel]] void UnaryOperation ## R <O ## Operator, T>(device T*, device const T*, uint);

#define FAST_RANKS S, V
#define FAST_OPS   Copy, Neg, Silu
#define FAST_TYPES uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, half, float, bfloat

INSTANTIATE3(FAST_FUNCTION, (FAST_RANKS), (FAST_OPS), (FAST_TYPES))

//
// Default unary opeator with source strides and matching rank
//

template <typename Op, typename T, typename U>
[[kernel]] void UnaryOperationRank1(device U* d,
                                   device const T* x,
                                   constant const size_t& stride_x,
                                   uint pos [[thread_position_in_grid]])
{
  auto idx_x = metal::pos_to_index(pos, stride_x);
  d[pos] = Op()(x[idx_x]);
}

template <typename Op, typename T, typename U>
[[kernel]] void UnaryOperationRank2(device U* d,
                                   device const T* x,
                                   constant const size_t strides_x[2],
                                   uint2 pos [[thread_position_in_grid]],
                                   uint2 grid_dim [[threads_per_grid]])
{
  auto idx_x = metal::pos_to_index(pos, strides_x);
  size_t c_idx = pos.x + (size_t)grid_dim.x * pos.y;
  d[c_idx] = Op()(x[idx_x]);
}

template <typename Op, typename T, typename U>
[[kernel]] void UnaryOperationRank3(device U* d,
                                   device const T* x,
                                   constant const size_t strides_x[3],
                                   uint3 pos [[thread_position_in_grid]],
                                   uint3 grid_dim [[threads_per_grid]])
{
  auto idx_x = metal::pos_to_index(pos, strides_x);
  size_t c_idx = pos.x + (size_t)grid_dim.x * (pos.y + (size_t)grid_dim.y * pos.z);
  d[c_idx] = Op()(x[idx_x]);
}


#define FULL_OPS   Copy, Neg, Silu
#define FULL_TYPES uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, half, float, bfloat

#define RANK1_FUNCTION(O, T) \
  template [[host_name(stringify(UnaryOperationRank1 ## O ## T))]]  \
  [[kernel]] void UnaryOperationRank1<O ## Operator, T, T>( \
    device T*, device const T*, \
    constant const size_t&, uint);

INSTANTIATE2(RANK1_FUNCTION, (FULL_OPS), (FULL_TYPES))

#define RANK2_FUNCTION(O, T) \
  template [[host_name(stringify(UnaryOperationRank2 ## O ## T))]]  \
  [[kernel]] void UnaryOperationRank2<O ## Operator, T, T>( \
    device T*, device const T*, \
    constant const size_t[2], uint2, uint2);

INSTANTIATE2(RANK2_FUNCTION, (FULL_OPS), (FULL_TYPES))

#define RANK3_FUNCTION(O, T) \
  template [[host_name(stringify(UnaryOperationRank3 ## O ## T))]]  \
  [[kernel]] void UnaryOperationRank3<O ## Operator, T, T>( \
    device T*, device const T*, \
    constant const size_t[3], uint3, uint3);

INSTANTIATE2(RANK3_FUNCTION, (FULL_OPS), (FULL_TYPES))
