//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_METAL_RMS_NORM_H
#define GRID_TENSOR_METAL_RMS_NORM_H

#include <math.h>
#include <tuple>

#include "binary.h"
#include "../precision.h"

namespace libai {

template <> class RmsNormOperator<device::Metal>
{
  // TODO: assumes "x" and "w" are contiguous
  template <typename T>
  void eval(MTL::Buffer* d, const MTL::Buffer* x, auto dimensions, auto strides_d, auto strides_x) const
  {
    const int n_reads = 4;

    auto& device = device::Metal::GetDevice();
    auto& encoder = device.Encoder();

    MTL::Size max_threads = device.max_threads_per_threadgroup_;
    unsigned long cols_limit = max_threads.width * n_reads;

    MTL::ComputePipelineState* pipeline;
    size_t n_cols = dimensions.back();
    size_t n_rows = std::accumulate(std::begin(dimensions), std::end(dimensions) - 1, 1, std::multiplies<size_t>());
    size_t n_threads = 0;

    if (n_cols > cols_limit)
    {
      static metal::Kernel<T> kernel("RmsNormLoop");
      pipeline = kernel.ComputePipelineState();
      n_threads = pipeline->maxTotalThreadsPerThreadgroup();
    }
    else
    {
      static metal::Kernel<T> kernel("RmsNormLine");
      pipeline = kernel.ComputePipelineState();
      size_t simd_width = pipeline->threadExecutionWidth();
      n_threads = ((n_cols + simd_width - 1) & -simd_width) / n_reads;
    }

    MTL::Size grid_size = MTL::Size(n_threads * n_rows, 1, 1);
    MTL::Size group_size = MTL::Size(n_threads, 1, 1);

    // TODO: assert (pipeline->maxTotalThreadsPerThreadgroup() > max_threads.width)

    encoder->setComputePipelineState(pipeline);

    T eps = Eps<T>::default_value;

    encoder->setBuffer(d, 0, 0);
    encoder->setBuffer(x, 0, 1);
    encoder->setBytes(&eps, sizeof(eps), 2);
    encoder->setBytes(&n_cols, sizeof(n_cols), 3);

    // TODO: hard-coded to float?
    encoder->setThreadgroupMemoryLength(pipeline->threadExecutionWidth() * sizeof(float), 0);
    encoder.DispatchThreads(grid_size, group_size);

    device.Wait();
  }

 public:

  // d = RMS(x) * x  -- Note: requires contiguous (not asserted)
  template<std::ranges::input_range I, std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<I>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<I>, std::ranges::iterator_t<O>>
  void operator()(I&& in, O&& out) const
  {
    using value_type = std::iter_value_t<std::ranges::iterator_t<O>>;

    auto first_d = std::ranges::begin(out);
    auto first_x = std::ranges::cbegin(in);

    std::span strides_d(first_d.Strides());
    std::span strides_x(first_x.Strides());

    eval<value_type>(first_d.Buffer(), first_x.Buffer(), first_x.Extents(), strides_d, strides_x);
  }
};

} // end of namespace libai

#endif  // GRID_TENSOR_METAL_RMS_NORM_H
