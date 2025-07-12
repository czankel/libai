//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/tensor.h>
#include <grid/tensor/precision.h>

#include <grid/tensor/cuda/device.h>
#include <grid/tensor/cuda/softmax.h>

#include "../instantiate.h"
#include "reduce.h"
#include "utils.h"

namespace grid {
namespace cuda {

template <typename T, unsigned int BlockSize>
__global__ void CudaSoftMax(T* d, const T* x, T eps, int dim)
{
  __shared__ T sdata[cuda::MaxThreadCount];

  unsigned int tid = threadIdx.x;
  T max{0};

  for (unsigned int i = tid; i < dim; i += BlockSize)
    max = x[i] > max ? x[i] : max;
  sdata[tid] = max;

  __syncthreads();

  CudaReduce<T, MaxOperator, BlockSize>(sdata, tid, dim);
  max = sdata[0];

  T sum{0};
  for (unsigned int i = tid; i < dim; i += BlockSize)
  {
    d[i] = exp(x[i] - max);
    sum += d[i];
  }
  sdata[tid] = sum;

  __syncthreads();

  CudaReduce<T, AddOperator, BlockSize>(sdata, tid, dim);
  sum = sdata[0];

  T scale = static_cast<T>(1)/(sum + eps);

  for (unsigned int i = tid; i < dim; i += BlockSize)
    d[i] = d[i] * scale;
}

} // end of namespace cuda


template <typename T>
void SoftMaxCallKernel(T* d, const T* x, size_t rows, size_t cols)
{
  size_t max_blocks = cuda::MaxThreadCount / sizeof(T);
  size_t dim_x = std::min(max_blocks, ((cols + 31) / 32) * 32);

  auto [grid_size, block_size] =
    cuda::GetSizes({rows, dim_x}, dim_x, 1);

  size_t n_threads = block_size.x;
  int n_threads_log2 = sizeof(n_threads) * 8 - __builtin_clzll(n_threads - 1);

  T eps = Eps<T>::default_value;
  switch (n_threads_log2)
  {
    #define CUDA_SOFTMAX_CASE(BIT) \
      case BIT: cuda::CudaSoftMax<T,1<<BIT><<<grid_size,block_size>>>(d,x,eps,cols); break;
    INSTANTIATE1(CUDA_SOFTMAX_CASE, (11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
    default: throw std::runtime_error(std::string("invalid thread count in softmax: ") +
                 std::to_string(n_threads_log2));
  }
}


// note that lower ranks are contiguous
template <typename T, size_t R>
void SoftMaxOperator<device::Cuda>::EvalContiguous(
    T* d, const T* x,
    std::span<const size_t, R> dimensions,
    std::span<const ssize_t, R> strides_d,
    std::span<const ssize_t, R> strides_a) const
{
  if constexpr (R == 0)
  {
    SoftMaxCallKernel(d, x, 1, 1);
  }
  else if constexpr (R == 1)
  {
    SoftMaxCallKernel(d, x, 1, dimensions[0]);
  }
  else if constexpr (R == 2)
  {
    throw std::runtime_error("softmax rank 2 not implemented");
  }
  else if constexpr (R == 3)
  {
    throw std::runtime_error("softmax rank 3 not implemented");
  }
}


template <typename T, size_t R>
void SoftMaxOperator<device::Cuda>::EvalDiscontiguous(
    T* d, const T* x,
    std::span<const size_t, R> dimensions,
    std::span<const ssize_t, R> strides_d,
    std::span<const ssize_t, R> strides_a) const
{
  if constexpr (R == 0)
  {
    SoftMaxCallKernel(d, x, 1, 1);
  }
  else
  { // if constexpr (R == 1)
    throw std::runtime_error("softmax on discontiguous tensors not implemented");
  }
}

#define FUNCTION_CONTIGUOUS(R, T) \
  template void SoftMaxOperator<device::Cuda>::EvalContiguous<T, R>( \
      T*, const T*, std::span<const size_t, R>, \
      std::span<const ssize_t, R>, std::span<const ssize_t, R>) const;

#define FUNCTION_DISCONTIGUOUS(R, T) \
  template void SoftMaxOperator<device::Cuda>::EvalDiscontiguous<T, R>( \
      T*, const T*,  std::span<const size_t, R>, \
      std::span<const ssize_t, R>, std::span<const ssize_t, R>) const;

#define TYPES  float
#define RANKS_CONTIGUOUS 1, 2, 3
#define RANKS_DISCONTIGUOUS 0, 1, 2, 3

INSTANTIATE2(FUNCTION_CONTIGUOUS, (RANKS_CONTIGUOUS), (TYPES))
INSTANTIATE2(FUNCTION_DISCONTIGUOUS, (RANKS_DISCONTIGUOUS), (TYPES))

} // end of namespace grid
