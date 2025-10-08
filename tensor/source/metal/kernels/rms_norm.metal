//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <metal_common>
#include <metal_math>
#include <metal_simdgroup>
#include <metal_stdlib>

#include "../../instantiate.h"

#include "utils.h"

#undef USE_REDUCE
#ifdef USE_REDUCE

template <typename T, int N_READS = 4>
[[kernel]] void RmsNorm(device T* d,
                        const device T* x,
                        constant float& eps,
                        constant uint& line_width,
                        threadgroup float* local_sums [[threadgroup(0)]],
                        uint gid [[threadgroup_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]],
                        __attribute__((unused)) uint group_size [[threads_per_threadgroup]],
                        uint simd_lane_id [[thread_index_in_simdgroup]],
                        uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
  threadgroup float sdata[metal::max_thread_];

  MetalReduce<T, AddOperator, BlockSize>(sdata, tid, row + dim);
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  if (simd_group_id == 0)
    sdata[row] = sqrt(sdata[row] / dim + eps);

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  for (unsigned int i = idx_beg; i < idx_end; i += grid_size)
    d[i] = x[i] / sdata[row];

}

#endif

template <typename T, int N_READS = 4>
[[kernel]] void RmsNormLine(device T* d,
                            const device T* x,
                            constant float& eps,
                            constant uint& line_width,
                            threadgroup float* local_sums [[threadgroup(0)]],
                            uint gid [[threadgroup_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]],
                            __attribute__((unused)) uint group_size [[threads_per_threadgroup]],
                            uint simd_lane_id [[thread_index_in_simdgroup]],
                            uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
  threadgroup float local_inv_mean;

  uint idx = gid * line_width + lid * N_READS;
  d += idx;
  x += idx;

  float acc = 0;
  if (lid * N_READS + N_READS <= line_width)
  {
    for (int i = 0; i < N_READS; i++)
    {
      float xi = x[i];
      acc += xi * xi;
    }
  }
  else
  {
    for (int i = 0; i < N_READS; i++)
    {
      if (lid * N_READS + i < line_width)
      {
        float xi = x[i];
        acc += xi * xi;
      }
    }
  }

  // Return the sum of the input values in acc across all active threads in the SIMD-group
  // and broadcasts the result to all active threads in the SIMD-group
  acc = metal::simd_sum(acc);

  // TODO: This assumes that the number of SIMD lanes is the same as the number of SIMD groups
  if (simd_group_id == 0)
    local_sums[simd_lane_id] = 0;

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  if (simd_lane_id == 0)
    local_sums[simd_group_id] = acc;

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  if (simd_group_id == 0)
  {
    acc = metal::simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0)
      local_inv_mean = metal::precise::rsqrt(acc / line_width + eps);
  }

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  if (lid * N_READS + N_READS <= line_width)
    for (int i = 0; i < N_READS; i++)
      d[i] = static_cast<T>(x[i] * local_inv_mean);
  else
    for (int i = 0; i < N_READS; i++)
      if (lid * N_READS + i < line_width)
        d[i] = static_cast<T>(x[i] * local_inv_mean);
}


template <typename T, int N_READS = 4>
[[kernel]] void RmsNormLoop(device T* d,
                            const device T* x,
                            constant float& eps,
                            constant uint& line_width,
                            threadgroup float* local_sums [[threadgroup(0)]],
                            uint gid [[threadgroup_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]],
                            uint group_size [[threads_per_threadgroup]],
                            uint simd_lane_id [[thread_index_in_simdgroup]],
                            uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
  threadgroup float local_inv_mean;

  uint idx = gid * line_width + lid * N_READS;
  d += idx;
  x += idx;

  float acc = 0; // TODO: use T or float?

  for (uint r = 0; r < line_width; r += group_size * N_READS)
    if (r + lid * N_READS + N_READS <= line_width)
      for (int i = 0; i < N_READS; i++)
      {
        float xi = x[i + r];
        acc += xi * xi;
      }
    else
      for (int i = 0; i < N_READS; i++)
        if ((r + lid * N_READS + i) < line_width)
        {
          float xi = x[i + r];
          acc += xi * xi;
        }

  acc = metal::simd_sum(acc);

  if (simd_group_id == 0)
    local_sums[simd_lane_id] = 0;

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  if (simd_lane_id == 0)
    local_sums[simd_group_id] = acc;

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  if (simd_group_id == 0)
  {
    acc = metal::simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0)
      local_inv_mean = metal::precise::rsqrt(acc / line_width + eps);
  }

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  for (uint r = 0; r < line_width; r += group_size * N_READS)
    if (r + lid * N_READS + N_READS <= line_width)
      for (int i = 0; i < N_READS; i++)
        d[r + i] = static_cast<T>(x[r + i] * local_inv_mean);
    else
      for (int i = 0; i < N_READS; i++)
        if ((r + lid * N_READS + i) < line_width)
          d[r + i] = static_cast<T>(x[r + i] * local_inv_mean);
}

#define RMS_NORM_OPS Line, Loop
#define RMS_NORM_TYPES uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, half, float, bfloat

#define RMS_NORM_FUNCTION(O, T) \
  template [[host_name(stringify(RmsNorm ## O ## T))]]  \
  [[kernel]] void RmsNorm ## O<T>( \
    device T*, \
    device const T*, \
    constant float&, \
    constant uint&, \
    threadgroup float*, \
    uint, uint, uint, uint, uint);

INSTANTIATE2(RMS_NORM_FUNCTION, (RMS_NORM_OPS), (RMS_NORM_TYPES))
