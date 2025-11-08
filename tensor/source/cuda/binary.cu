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

#include "../instantiate.h"
#include "utils.h"

namespace libai {

// FIXME: note : Similar to thread blocks, clusters are also organized into a one-dimension, two-dimension, or three-dimension as illustrated by Figure 5. The number of thread blocks in a cluster can be user-defined, and a maximum of 8 thread blocks in a cluster is supported as a portable cluster size in CUDA. Note that on GPU hardware or MIG configurations which are too small to support 8 multiprocessors the maximum cluster size will be reduced accordingly. Identification of these smaller configurations, as well as of larger configurations supporting a thread block cluster size beyond 8, is architecture-specific and can be queried using the cudaOccupancyMaxPotentialClusterSize API.

// FIXME: use __restrict__ where possible
// __builtin_assume_aligned
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=strides#occupancy-calculator

//
// Binary Operators
//

template <> struct AddOperator<device::Cuda>
{ template<typename T> inline __device__ T operator()(T a, T b) { return a + b; } };
template <> struct SubOperator<device::Cuda>
{ template<typename T> inline __device__ T operator()(T a, T b) { return a - b; } };
template <> struct MulOperator<device::Cuda>
{ template<typename T> inline __device__ T operator()(T a, T b) { return a * b; } };
template <> struct DivOperator<device::Cuda>
{ template<typename T> inline __device__ T operator()(T a, T b) { return a / b; } };

//
// Kernels
//

// TODO: there are many different kernels (too many?), are they all needed?
// TODO: can this be optimzied, i.e. with vector instructions or other HW component?
// FIXME: overflow for int??
template <template <typename> typename O, typename T>
__global__ void CudaBinarySS(T* d, const T* a, const T* b)
{
  d[0] = O<device::Cuda>()(a[0], b[0]);
}

template <template <typename> typename O, typename T>
__global__ void CudaBinaryVS(T* d, const T* a, const T* b, size_t n)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n)
    d[index] = O<device::Cuda>()(a[index], b[0]);
}

template <template <typename> typename O, typename T>
__global__ void CudaBinarySV(T* d, const T* a, const T* b, size_t n)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n)
    d[index] = O<device::Cuda>()(a[0], b[index]);
}

template <template <typename> typename O, typename T>
__global__ void CudaBinaryVV(T* d, const T* a, const T* b, size_t n)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n)
    d[index] = O<device::Cuda>()(a[index], b[index]);
}

template <template <typename> typename O, typename T>
__global__ void CudaBinaryContiguousRank2(
    T* d, const T* a, const T* b, dim3 dims, dim3 strides_d, dim3 strides_a, dim3 strides_b)
{
  size_t idx_x = blockIdx.x * blockDim.x + threadIdx.x; // is there a threadIdx.y???
  if (idx_x < dims.x)
  {
    size_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_y < dims.y)
    {
      size_t idx_a = idx_y * strides_a.y + idx_x;
      size_t idx_b = idx_y * strides_b.y + idx_x;
      size_t idx_d = idx_y * strides_d.y + idx_x;
      d[idx_d] = O<device::Cuda>()(a[idx_a], b[idx_b]);
    }
  }
}

template <template <typename> typename O, typename T>
__global__ void CudaBinaryContiguousRank3(
    T* d, const T* a, const T* b, dim3 dims, dim3 strides_d, dim3 strides_a, dim3 strides_b)
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
        size_t idx_a = idx_z * strides_a.z + idx_y * strides_a.y + idx_x;
        size_t idx_b = idx_z * strides_b.z + idx_y * strides_b.y + idx_x;
        size_t idx_d = idx_z * strides_d.z + idx_y * strides_d.y + idx_x;
        d[idx_d] = O<device::Cuda>()(a[idx_a], b[idx_b]);
      }
    }
  }
}

template <template <typename> typename O, typename T>
__global__ void CudaBinaryDiscontiguousRank1(
    T* d, const T* a, const T* b, size_t dim, size_t stride_d, size_t stride_a, size_t stride_b)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dim)
  {
    int idx_a = idx * stride_a;
    int idx_b = idx * stride_b;
    int idx_d = idx * stride_d;
    d[idx_d] = O<device::Cuda>()(a[idx_a], b[idx_b]);
  }
}

template <template <typename> typename O, typename T>
__global__ void CudaBinaryDiscontiguousRank2(
    T* d, const T* a, const T* b, dim3 dims, dim3 strides_d, dim3 strides_a, dim3 strides_b)
{
  size_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx_x < dims.x)
  {
    size_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_y < dims.y)
    {
      size_t idx_a = idx_y * strides_a.y + idx_x * strides_a.x;
      size_t idx_b = idx_y * strides_b.y + idx_x * strides_b.x;
      size_t idx_d = idx_y * strides_d.y + idx_x * strides_d.x;
      d[idx_d] = O<device::Cuda>()(a[idx_a], b[idx_b]);
    }
  }
}

template <template <typename> typename O, typename T>
__global__ void CudaBinaryDiscontiguousRank3(
    T* d, const T* a, const T* b, dim3 dims, dim3 strides_d, dim3 strides_a, dim3 strides_b)
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
        size_t idx_b = idx_z * strides_b.z + idx_y * strides_b.y + idx_x * strides_b.x;
        size_t idx_d = idx_z * strides_d.z + idx_y * strides_d.y + idx_x * strides_d.x;
        d[idx_d] = O<device::Cuda>()(a[idx_a], b[idx_b]);
      }
    }
  }
}

//
// Eval* Definitions
//

template <template <typename> typename O>
template <typename T>
void BinaryOperation<O, device::Cuda>::EvalSS(T* d, const T* a, const T* b, size_t) const
{
  CudaBinarySS<O, T><<<1, 1>>>(d, a, b);
}

template <template <typename> typename O>
template <typename T>
void BinaryOperation<O, device::Cuda>::EvalSV(T* d, const T* a, const T* b, size_t size) const
{
  auto [grid_size, block_size] = cuda::GetSizes(size);
  CudaBinarySV<O, T><<<grid_size, block_size>>>(d, a, b, size);
}

template <template <typename> typename O>
template <typename T>
void BinaryOperation<O, device::Cuda>::EvalVS(T* d, const T* a, const T* b, size_t size) const
{
  auto [grid_size, block_size] = cuda::GetSizes(size);
  CudaBinaryVS<O, T><<<grid_size, block_size>>>(d, a, b, size);
}

template <template <typename> typename O>
template <typename T>
void BinaryOperation<O, device::Cuda>::EvalVV(T* d, const T* a, const T* b, size_t size) const
{
  auto [grid_size, block_size] = cuda::GetSizes(size);
  CudaBinaryVV<O, T><<<grid_size, block_size>>>(d, a, b, size);
}

// note that lower ranks are contiguous
template <template <typename> typename O>
template <typename T, size_t R>
void BinaryOperation<O, device::Cuda>::EvalContiguous(
    T* d, const T* a, const T* b, std::span<const size_t, R> dimensions,
    std::span<const ssize_t, R> strides_d,
    std::span<const ssize_t, R> strides_a,
    std::span<const ssize_t, R> strides_b) const
{
  if constexpr (R == 2)
  {
    auto [block_size, grid_size] = cuda::GetSizes(dimensions, 16, 16);  // FIXME 256 threads instead of 1k? 32, 32?
    CudaBinaryContiguousRank2<O, T><<<block_size, grid_size>>>(
        d, a, b,
        cuda::MakeDim3(dimensions),
        cuda::MakeDim3(strides_d), cuda::MakeDim3(strides_a), cuda::MakeDim3(strides_b));
  }
  else if constexpr (R == 3)
  {
    auto [block_size, grid_size] = cuda::GetSizes(dimensions, 8, 8, 8);   // FIXME 512 threads?
    CudaBinaryContiguousRank3<O, T><<<block_size, grid_size>>>(
        d, a, b,
        cuda::MakeDim3(dimensions),
        cuda::MakeDim3(strides_d),
        cuda::MakeDim3(strides_a),
        cuda::MakeDim3(strides_b));
  }
}

template <template <typename> typename O>
template <typename T, size_t R>
void BinaryOperation<O, device::Cuda>::EvalDiscontiguous(
    T* d, const T* a, const T* b, std::span<const size_t, R> dimensions,
    std::span<const ssize_t, R> strides_d,
    std::span<const ssize_t, R> strides_a,
    std::span<const ssize_t, R> strides_b) const
{
  if constexpr (R == 1)
  {
    auto [grid_size, block_size] = cuda::GetSizes(dimensions[0]);
    CudaBinaryDiscontiguousRank1<O, T><<<block_size, grid_size>>>(
        d, a, b, dimensions[0], strides_d[0], strides_a[0], strides_b[0]);
  }
  else if constexpr (R == 2)
  {
    auto [block_size, grid_size] = cuda::GetSizes(dimensions, 16, 16);  // FIXME 256 threads instead of 1k? 32, 32?
    CudaBinaryDiscontiguousRank2<O, T><<<block_size, grid_size>>>(
        d, a, b,
        cuda::MakeDim3(dimensions),
        cuda::MakeDim3(strides_d), cuda::MakeDim3(strides_a), cuda::MakeDim3(strides_b));
  }
  else if constexpr (R == 3)
  {
    auto [block_size, grid_size] = cuda::GetSizes(dimensions, 8, 8, 8);   // FIXME 512 threads?
    CudaBinaryDiscontiguousRank3<O, T><<<block_size, grid_size>>>(
        d, a, b,
        cuda::MakeDim3(dimensions),
        cuda::MakeDim3(strides_d), cuda::MakeDim3(strides_a), cuda::MakeDim3(strides_b));
  }
}

// Instantiate the Eval methods for all supported types and operations
// This is necessary to use the NVCC compiler only for the CU files (for now).

#define OPS    Add, Sub, Mul, Div
#define TYPES  int, float

#define FUNCTIONS_VECSCALAR(R, O, T) \
  template void BinaryOperation<O ##Operator, device::Cuda>::Eval##R<T>( \
      T*, const T*, const T*, size_t) const;

#define QUANTITIES  SS,SV,VS,VV

INSTANTIATE3(FUNCTIONS_VECSCALAR, (QUANTITIES), (OPS), (TYPES))

  // FIXME: why span if contiguous??
#define FUNCTION_CONTIGUOUS(R, O, T) \
  template void BinaryOperation<O ##Operator, device::Cuda>::EvalContiguous<T, R>( \
      T*, const T*, const T*, std::span<const size_t, R>, \
      std::span<const ssize_t, R>, std::span<const ssize_t, R>, std::span<const ssize_t, R>) const; 

#define FUNCTION_DISCONTIGUOUS(R, O, T) \
  template void BinaryOperation<O ##Operator, device::Cuda>::EvalDiscontiguous<T, R>( \
      T*, const T*, const T*, std::span<const size_t, R>, \
      std::span<const ssize_t, R>, std::span<const ssize_t, R>, std::span<const ssize_t, R>) const;

#define RANKS_CONTIGUOUS 2, 3
#define RANKS_DISCONTIGUOUS 1, 2, 3

INSTANTIATE3(FUNCTION_CONTIGUOUS, (RANKS_CONTIGUOUS), (OPS), (TYPES))
INSTANTIATE3(FUNCTION_DISCONTIGUOUS, (RANKS_DISCONTIGUOUS), (OPS), (TYPES))

} // end of namespace libai
