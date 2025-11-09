//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <utility>

#include <libai/tensor/tensor.h>

#include <libai/tensor/cuda/binary.h>
#include <libai/tensor/cuda/device.h>

#include <libai/tensor/cuda/rms_norm.h>

#include "../instantiate.h"
#include "reduce.h"
#include "utils.h"


namespace libai {
namespace cuda {

// TODO support strides
// only operate on 1 block in x direction and N in y
template <typename T, unsigned int BlockSize>
__global__ void CudaRmsNorm(T* d, const T* x, const T eps, int dim)
{
  __shared__ T sdata[cuda::MaxThreadCount];

  int grid_size = gridDim.x * BlockSize;
  int row = threadIdx.y * blockDim.x;
  int tid = row + threadIdx.x;

  size_t idx_first = (blockIdx.y * blockDim.y + threadIdx.y) * dim;
  size_t idx_beg = idx_first + tid;
  size_t idx_end = idx_first + dim;

  T sum{0};
  for (size_t i = idx_beg; i < idx_end; i += grid_size)
    sum += x[i] * x[i];
  sdata[tid] = sum;

  __syncthreads();

  CudaReduce<T, AddOperator, BlockSize>(sdata, tid, row + dim);

  if (threadIdx.x == 0)
    sdata[row] = sqrt(sdata[row] / dim + eps);

  __syncthreads();

  for (unsigned int i = idx_beg; i < idx_end; i += grid_size)
    d[i] = x[i] / sdata[row];
}

} // end of namespace cuda

template <typename T>
void
RmsNormOperator<device::Cuda>::Eval(T* d, const T* x, const T eps, size_t dim_y, size_t dim_x) const
{
  // make blocks of [1024 / sizeof(T) / 2, 1]:
  //  - N columns: limit to fit into a single 1024k page
  //  - 1 row: parallelize operation

  size_t threads = std::min(dim_x, cuda::MaxThreadCount / sizeof(T) / 2);
  auto [grid_size, block_size] = cuda::GetSizes({threads, dim_y}, threads, 1);

  size_t smem_size = cuda::MaxThreadCount;
  size_t n_threads = grid_size.x;
  int n_threads_log2 = sizeof(n_threads) * 8 - __builtin_clzll(n_threads - 1);

  switch (n_threads_log2)
  {
    #define CUDA_RMS_NORM_CASE(BIT) \
      case BIT: cuda::CudaRmsNorm<T,1<<BIT><<<grid_size,block_size,smem_size>>>(d,x,eps,dim_x); break;
    INSTANTIATE1(CUDA_RMS_NORM_CASE, (10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
    default: throw std::runtime_error("invalid thread count");
  }
}

#define FUNCTION(T) \
  template void RmsNormOperator<device::Cuda>::Eval<T>(T*, const T*, const T, size_t, size_t) const; 

#define TYPES  float

INSTANTIATE1(FUNCTION, (TYPES))

} // end of namespace libai
