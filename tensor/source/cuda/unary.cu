//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <libai/tensor//tensor.h>
#include <libai/tensor/cuda/device.h>
#include <libai/tensor/cuda/unary.h>

#include "../instantiate.h"
#include "utils.h"

namespace libai {

//
// Elementary Unary Operators
//

template <> struct libai::CopyOperator<libai::device::Cuda>
{ template<typename T> inline __device__ T operator()(const T x) const { return x; } };
template <> struct libai::NegOperator<libai::device::Cuda>
{ template<typename T> inline __device__ T operator()(const T x) const { return -x; } };

//
// Unary Functions
//

template <> struct libai::SiluFunction<libai::device::Cuda>
{
  template<typename T> inline __device__ T operator()(const T x) const { return x / (T{1} + expf(-x)); }
};

//
// Kernels
//

template <template <typename> typename O, typename T>
__global__ void CudaUnaryScalar(T* c, const T* a)
{
  c[0] = O<libai::device::Cuda>()(a[0]);
}

template <template <typename> typename O, typename T>
__global__ void CudaUnaryVector(T* c, const T* a, size_t n)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n)
    c[index] = O<libai::device::Cuda>()(a[index]);
}

template <template <typename> typename O, typename T>
__global__ void CudaUnaryContiguousRank2(
    T* d, const T* a, dim3 dims, dim3 strides_d, dim3 strides_a)
{
  size_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx_x < dims.x)
  {
    size_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_y < dims.y)
    {
      size_t idx_a = idx_y * strides_a.y + idx_x;
      size_t idx_d = idx_y * strides_d.y + idx_x;
      d[idx_d] = O<device::Cuda>()(a[idx_a]);
    }
  }
}

template <template <typename> typename O, typename T>
__global__ void CudaUnaryContiguousRank3(T* d, const T* a, dim3 dims, dim3 strides_d, dim3 strides_a)
{
  size_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx_x < dims.x)
  {
    size_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_y < dims.y)
    {
      size_t idx_z = blockIdx.z * blockDim.z + threadIdx.y;
      if (idx_z < dims.z)
      {
        size_t idx_a = idx_z * strides_a.z + idx_y * strides_a.y + idx_x;
        size_t idx_d = idx_z * strides_d.z + idx_y * strides_d.y + idx_x;
        d[idx_d] = O<device::Cuda>()(a[idx_a]);
      }
    }
  }
}

template <template <typename> typename O, typename T>
__global__ void CudaUnaryDiscontiguousRank1(
    T* d, const T* a, size_t dim, size_t stride_d, size_t stride_a)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dim)
  {
    int idx_a = idx * stride_a;
    int idx_d = idx * stride_d;
    d[idx_d] = O<device::Cuda>()(a[idx_a]);
  }
}

template <template <typename> typename O, typename T>
__global__ void CudaUnaryDiscontiguousRank2(
    T* d, const T* a, dim3 dims, dim3 strides_d, dim3 strides_a)
{
  size_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx_x < dims.x)
  {
    size_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_y < dims.y)
    {
      size_t idx_a = idx_y * strides_a.y + idx_x * strides_a.x;
      size_t idx_d = idx_y * strides_d.y + idx_x * strides_d.x;
      d[idx_d] = O<device::Cuda>()(a[idx_a]);
    }
  }
}

template <template <typename> typename O, typename T>
__global__ void CudaUnaryDiscontiguousRank3(
    T* d, const T* a, dim3 dims, dim3 strides_d, dim3 strides_a)
{
  size_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx_x < dims.x)
  {
    size_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_y < dims.y)
    {
      size_t idx_z = blockIdx.z * blockDim.z + threadIdx.z;
      if (idx_z < dims.z)
      {
        size_t idx_a = idx_z * strides_a.z + idx_y * strides_a.y + idx_x * strides_a.x;
        size_t idx_d = idx_z * strides_d.z + idx_y * strides_d.y + idx_x * strides_d.x;
        d[idx_d] = O<device::Cuda>()(a[idx_a]);
      }
    }
  }
}

//
// Eval* Definitions
//

// note that lower ranks are contiguous
template <template <typename> typename O>
template <typename T, size_t R>
void UnaryOperation<O, device::Cuda>::EvalContiguous(
    T* d, const T* a,
    std::span<const size_t, R> dimensions,
    std::span<const ssize_t, R> strides_d,
    std::span<const ssize_t, R> strides_a) const
{
  if constexpr (R == 0)
  {
    CudaUnaryScalar<O, T><<<1, 1>>>(d, a);
  }
  else if constexpr (R == 1)
  {
    auto [grid_size, block_size] = cuda::GetSizes(dimensions[0]);
    CudaUnaryVector<O, T><<<grid_size, block_size>>>(d, a, dimensions[0]);
  }
  else if constexpr (R == 2)
  {
    auto [block_size, grid_size] = cuda::GetSizes(dimensions, 16, 16);  // FIXME 256 threads instead of 1k? 32, 32?
    CudaUnaryContiguousRank2<O, T><<<block_size, grid_size>>>(
        d, a, cuda::MakeDim3(dimensions), cuda::MakeDim3(strides_d), cuda::MakeDim3(strides_a));
  }
  else if constexpr (R == 3)
  {
    auto [block_size, grid_size] = cuda::GetSizes(dimensions, 8, 8, 8);   // FIXME 512 threads?
    CudaUnaryContiguousRank3<O, T><<<block_size, grid_size>>>(
        d, a, cuda::MakeDim3(dimensions), cuda::MakeDim3(strides_d), cuda::MakeDim3(strides_a));
  }
}


template <template <typename> typename O>
template <typename T, size_t R>
void UnaryOperation<O, device::Cuda>::EvalDiscontiguous(
    T* d, const T* a,
    std::span<const size_t, R> dimensions,
    std::span<const ssize_t, R> strides_d,
    std::span<const ssize_t, R> strides_a) const
{
  if constexpr (R == 0)
  {
    CudaUnaryScalar<O, T><<<1, 1>>>(d, a);
  }
  else if constexpr (R == 1)
  {
    auto [grid_size, block_size] = cuda::GetSizes(dimensions[0]);
    CudaUnaryDiscontiguousRank1<O, T><<<block_size, grid_size>>>(
        d, a, dimensions[0], strides_d[0], strides_a[0]);
  }
  else if constexpr (R == 2)
  {
    auto [block_size, grid_size] = cuda::GetSizes(dimensions, 16, 16);  // FIXME 256 threads instead of 1k? 32, 32?
    CudaUnaryDiscontiguousRank2<O, T><<<block_size, grid_size>>>(
        d, a, cuda::MakeDim3(dimensions), cuda::MakeDim3(strides_d), cuda::MakeDim3(strides_a));
  }
  else if constexpr (R == 3)
  {
    auto [block_size, grid_size] = cuda::GetSizes(dimensions, 8, 8, 8);   // FIXME 512 threads?
    CudaUnaryDiscontiguousRank3<O, T><<<block_size, grid_size>>>(
        d, a, cuda::MakeDim3(dimensions), cuda::MakeDim3(strides_d), cuda::MakeDim3(strides_a));
  }
}

#define FUNCTION_CONTIGUOUS(R, O, T) \
  template void UnaryOperation<O, device::Cuda>::EvalContiguous<T, R>( \
      T*, const T*, std::span<const size_t, R>, \
      std::span<const ssize_t, R>, std::span<const ssize_t, R>) const;

#define FUNCTION_DISCONTIGUOUS(R, O, T) \
  template void UnaryOperation<O, device::Cuda>::EvalDiscontiguous<T, R>( \
      T*, const T*, std::span<const size_t, R>, \
      std::span<const ssize_t, R>, std::span<const ssize_t, R>) const;

#define OPS    CopyOperator, NegOperator, SiluFunction
#define TYPES  float,int
#define RANKS_CONTIGUOUS 1, 2, 3
#define RANKS_DISCONTIGUOUS 0, 1, 2, 3

INSTANTIATE3(FUNCTION_CONTIGUOUS, (RANKS_CONTIGUOUS), (OPS), (TYPES))
INSTANTIATE3(FUNCTION_DISCONTIGUOUS, (RANKS_DISCONTIGUOUS), (OPS), (TYPES))

} // end of namespace libai
