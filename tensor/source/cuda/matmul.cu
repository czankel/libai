//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <utility>

#include <grid/tensor/tensor.h>

#include <grid/tensor/cuda/device.h>
#include <grid/tensor/cuda/matmul.h>

#include "../instantiate.h"
#include "utils.h"

namespace libai {

// Note that dimensions are jki  (col, index, row)
template <typename T>
__global__ void CudaMatmulGeneric(T* d, const T* a, const T* b,
                                  dim3 dims,
                                  dim3 strides_d, dim3 strides_a, dim3 strides_b)
{
  size_t idx_i = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx_i < dims.z)
  {
    size_t idx_j = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_j < dims.x)
    {
      size_t idx_d = idx_i * strides_d.y + idx_j * strides_d.x;
      size_t idx_a = idx_i * strides_a.y;
      size_t idx_b = idx_j * strides_b.x;

      T sum{0};
      for (size_t idx_k = 0; idx_k < dims.y; idx_k++)
        sum += a[idx_a + idx_k * strides_a.x] * b[idx_b + idx_k * strides_b.y];
      d[idx_d] = sum;
    }
  }
}

template <typename T>
void MatmulOperator<device::Cuda>::EvalGeneric(
    T* d, const T* a, const T* b,
    std::span<const size_t, 3> dimensions,
    std::span<const ssize_t, 2> strides_d,
    std::span<const ssize_t, 2> strides_a,
    std::span<const ssize_t, 2> strides_b) const
{
  auto [grid_size, block_size] = cuda::GetSizes({dimensions[0], dimensions[2]}, 16, 16, 1);
  CudaMatmulGeneric<T><<<grid_size, block_size>>>(
      d, a, b,
      cuda::MakeDim3(dimensions),
      cuda::MakeDim3(strides_d), cuda::MakeDim3(strides_a), cuda::MakeDim3(strides_b));
}

#define FUNCTION(T) \
  template void MatmulOperator<device::Cuda>::EvalGeneric<T>( \
      T*, const T*, const T*, \
      std::span<const size_t, 3>, \
      std::span<const ssize_t, 2>, std::span<const ssize_t, 2>, std::span<const ssize_t, 2>) const;

#define TYPES  int, float

INSTANTIATE1(FUNCTION, (TYPES))

} // end of namespace libai
