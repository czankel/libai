//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef LIBAI_TENSOR_METAL_MATMUL_H
#define LIBAI_TENSOR_METAL_MATMUL_H

#include "grid/util/demangle.h"

#include "device.h"
#include "kernels.h"

namespace libai {

/// MatmulOperator implements a multiplication operation for tensors
/// different ranks, such as matrix multiplication (Matmul) and vector dot-product (VecDot).
template <> class MatmulOperator<device::Metal>
{
  // FIXME: fix order of dst vs src across all files!!
  template <typename T>
  inline void eval(MTL::Buffer* d, const MTL::Buffer* x, const MTL::Buffer* y,
                   std::span<const size_t,  2> dimensions,
                   size_t dim_j,
                   std::span<const ssize_t, 2> strides_d,
                   std::span<const ssize_t, 2> strides_x,
                   std::span<const ssize_t, 2> strides_y) const
  {
    auto& device = device::Metal::GetDevice();
    auto& encoder = device.Encoder();

    encoder->setBuffer(d, 0, 0);
    encoder->setBuffer(x, 0, 1);
    encoder->setBuffer(y, 0, 2);

    encoder->setBytes(&dim_j, sizeof(size_t), 3);
    encoder->setBytes(strides_d.data(), strides_d.size() * sizeof(size_t), 4);
    encoder->setBytes(strides_x.data(), strides_x.size() * sizeof(size_t), 5);
    encoder->setBytes(strides_y.data(), strides_y.size() * sizeof(size_t), 6);

    static metal::Kernel<T> kernel("GeMMOperatorGeneric");
    MTL::ComputePipelineState* pipeline = kernel.ComputePipelineState();
    encoder->setComputePipelineState(pipeline);

    /*
    NS::UInteger thread_group_size_ = std::min(dimensions[2], kernel_->maxTotalThreadsPerThreadgroup());
    MTL::Size thread_group_size = MTL::Size(thread_group_size_, 1, 1);
    enc->dispatchThreads(grid_size, thread_group_size);
    */

    // FIXME MTL::Size grid_size = MTL::Size(dimensions[0], dimensions[1], 1);
    // FIXME: assumes contiguous?
    size_t array_length = /*strides_d[0] * */ dimensions[0];
    MTL::Size grid_size = MTL::Size(array_length, dimensions[1], 1);

    NS::UInteger thread_group_size = std::min(array_length, pipeline->maxTotalThreadsPerThreadgroup());
    MTL::Size group_size = MTL::Size(thread_group_size, 1, 1);

    encoder.DispatchThreads(grid_size, group_size);

    device.Wait();
  }

 public:
  // TODO: this is a generic unoptimized implementation
  template<std::ranges::input_range I1,
           std::ranges::input_range I2,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<I1>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<I1>, std::ranges::iterator_t<O>> &&
           std::indirectly_copyable<std::ranges::iterator_t<I2>, std::ranges::iterator_t<O>>
  void operator()(I1&& in1, I2&& in2, O&& out) const
  {
    using value_type = std::iter_value_t<std::ranges::iterator_t<O>>;

    auto first_d = std::ranges::begin(out);
    auto first_x = std::ranges::cbegin(in1);
    auto first_y = std::ranges::cbegin(in2);

    constexpr size_t rank_x = std::ranges::iterator_t<I1>::rank;
    constexpr size_t rank_y = std::ranges::iterator_t<I2>::rank;

    // FIXME: vecdot has rank 0!!!
    size_t dim_j = first_x.Extents()[rank_x - 1];

    if (dim_j != first_y.Extents()[0])
      throw std::runtime_error("GeMM: inner dimensions don't match");

    if constexpr (rank_x == 2 && rank_y == 2)
      eval<value_type>(first_d.Buffer(), first_x.Buffer(), first_y.Buffer(),
                       std::span(first_x.Extents()),
                       dim_j,
                       std::span(first_d.Strides()),
                       std::span(first_x.Strides()),
                       std::span(first_y.Strides()));
    else if constexpr (rank_x == 2 && rank_y == 1)
      eval<value_type>(first_d.Buffer(), first_x.Buffer(), first_y.Buffer(),
                       std::array<size_t, 2>{first_x.Extents()[0], 1},
                       dim_j,
                       std::array<const ssize_t, 2>{first_d.Strides()[0], 0},
                       std::span(first_x.Strides()),
                       std::array<const ssize_t, 2>{first_y.Strides()[0], 0});
    else if constexpr (rank_x == 1 && rank_y == 2)
      eval<value_type>(first_d.Buffer(), first_x.Buffer(), first_y.Buffer(),
                       std::array<size_t, 2>{1, first_y.Extents()[1]},
                       dim_j,
                       std::array<const ssize_t, 2>{0, first_d.Strides()[0]},
                       std::array<const ssize_t, 2>{0, first_x.Strides()[0]},
                       std::span(first_y.Strides()));
    else if constexpr (rank_x == 1 && rank_y == 1)
      eval<value_type>(first_d.Buffer(), first_x.Buffer(), first_y.Buffer(),
                       std::array<size_t, 2>{1, 1},
                       dim_j,
                       std::array<const ssize_t, 2>{0, 0},
                       std::array<const ssize_t, 2>{0, first_x.Strides()[0]},
                       std::array<const ssize_t, 2>{first_y.Strides()[0], 0});
    else
      throw std::runtime_error("invalid matrix/vector multiplication");
  }
};

} // end of namespace libai

#endif  // LIBAI_TENSOR_METAL_MATMUL_H
