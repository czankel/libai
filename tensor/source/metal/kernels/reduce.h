//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef TENSOR_SOURCE_METAL_REDUCE_H
#define TENSOR_SOURCE_METAL_REDUCE_H

namespace grid {
namespace metal {

template <>
void MetalSIMDReduce()
{
  acc = metal::simd_sum(acc);
}

template <typename T, typename O, unsigned int BlockSize>
inline void MetalReduce(volatile T* sdata, unsigned int )
{
  if (BlockSize >= 512)
  {
    if (tid < 256) { sdata[tid] = O()(sdata[tid], sdata[tid + 256]); }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }
  if (BlockSize >= 256)
  {
    if (tid < 128) { sdata[tid] = O()(sdata[tid], sdata[tid + 128]); }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }
  if (BlockSize >= 128)
  {
    if (tid < 64)  { sdata[tid] = O()(sdata[tid], sdata[tid + 64]);  }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }
  if (tid < 32) // FIXME this would seem to be the SIMD width?
    WarpReduce<T, O, BlockSize>(sdata, tid, dim);
  // FIXME or this?
  xx = metal::simd_sum(xx);

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
}


} // end of namespace metal
} // end of namespace grid

#endif  // TENSOR_SOURCE_METAL_REDUCE_H
