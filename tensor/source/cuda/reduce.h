//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef TENSOR_SOURCE_CUDA_REDUCE_H 
#define TENSOR_SOURCE_CUDA_REDUCE_H 

// TODO: This is just a placeholder, and it might not work on more recent devices
// TODO: See also: https://github.com/ashvardanian/ParallelReductionsBenchmark?tab=readme-ov-file

namespace libai {
namespace cuda {

struct AddOperator
{
  template <typename T>
  __device__ inline T operator()(const T& x, const T& y) const { return x + y; }
};

struct MaxOperator
{
  template <typename T>
  __device__ inline T operator()(const T& x, const T& y) const { return x > y ? x : y; }
};

// TODO: align to warp sizes

// Reduce two consecutive warps to a single warp by applying the operation.
// tid_x is the thread x-index in the current block and line, tid_e the last index for that block.
template <typename T, typename O, unsigned int BlockSize>
__device__ inline
void CudaWarpReduce(volatile T* sdata, unsigned int tid, unsigned int dim)
{
  if (BlockSize >= 64 && tid + 32 < dim) sdata[tid] = O()(sdata[tid], sdata[tid + 32]);
  if (BlockSize >= 32 && tid + 16 < dim) sdata[tid] = O()(sdata[tid], sdata[tid + 16]);
  if (BlockSize >= 16 && tid +  8 < dim) sdata[tid] = O()(sdata[tid], sdata[tid +  8]);
  if (BlockSize >=  8 && tid +  4 < dim) sdata[tid] = O()(sdata[tid], sdata[tid +  4]);
  if (BlockSize >=  4 && tid +  2 < dim) sdata[tid] = O()(sdata[tid], sdata[tid +  2]);
  if (BlockSize >=  2 && tid +  1 < dim) sdata[tid] = O()(sdata[tid], sdata[tid +  1]);
}

// Reduce a block along the x-axis for a thread with index tid in the current block.
// idx_x is the x-index of the thread in the block and line, and tid_e last index in that block
// Note that due to memory bandwidth and layout (cache) constraints should limit nr of threads
template <typename T, typename O, unsigned int BlockSize>
__device__ inline
void CudaReduce(volatile T* sdata, unsigned int tid, unsigned int dim)
{
  if (BlockSize >= 512)
  {
    if (tid < 256) { sdata[tid] = O()(sdata[tid], sdata[tid + 256]); } __syncthreads();
  }
  if (BlockSize >= 256)
  {
    if (tid < 128) { sdata[tid] = O()(sdata[tid], sdata[tid + 128]); } __syncthreads();
  }
  if (BlockSize >= 128)
  {
    if (tid < 64)  { sdata[tid] = O()(sdata[tid], sdata[tid + 64]);  } __syncthreads();
  }
  if (tid < 32)
    CudaWarpReduce<T, O, BlockSize>(sdata, tid, dim);

  __syncthreads();
}

} // end of namespace cuda
} // end of namespace libai


#endif  // TENSOR_SOURCE_CUDA_REDUCE_H 
