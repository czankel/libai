//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef LIBAI_TENSOR_METAL_UNARY_H
#define LIBAI_TENSOR_METAL_UNARY_H

#include <libai/util/demangle.h>

#include "device.h"
#include "kernels.h"
#include "utils.h"

namespace libai {

// UnaryOperation<Operator> implements element-wise unary operation on a tensors for metal devices.
template <template <typename> typename TOperator>
class UnaryOperation<TOperator, device::Metal>
{
  template <typename T>
  void Eval(MTL::Buffer* d_buf, const MTL::Buffer* x_buf,
            size_t d_ofs, size_t x_ofs,
            auto dimensions, auto strides_d, auto strides_x) const
  {
    constexpr size_t rank = dimensions.size();

    auto& device = device::Metal::GetDevice();
    auto& encoder = device.Encoder();

    encoder->setBuffer(d_buf, d_ofs, 0);
    encoder->setBuffer(x_buf, x_ofs, 1);

    size_t s1 = strides_x.size();

    MTL::ComputePipelineState* pipeline;
    if (rank == 0 || (rank == 1 && (s1 == 0 || strides_x[s1 - 1] == 1)))
    {
      std::string quantity = s1 == 0 ? "S" : "V";
      static metal::Kernel<T> kernel("UnaryOperation" + quantity + std::string(TOperator<device::Metal>::kernel_name));

      pipeline = kernel.ComputePipelineState();
      encoder->setComputePipelineState(pipeline);

      size_t array_length = 1;
      if constexpr (rank > 0)
      {
        array_length = dimensions[0];
        if (strides_d.size() != 0)
          array_length *= strides_d[0];
      }

      MTL::Size grid_size = MTL::Size(array_length, 1, 1);
      NS::UInteger thread_group_size_ = std::min(array_length, pipeline->maxTotalThreadsPerThreadgroup());
      MTL::Size thread_group_size = MTL::Size(thread_group_size_, 1, 1);

      encoder.DispatchThreads(grid_size, thread_group_size);

      device.Wait(); // TODO: use callback or manage dispaltched jobs
    }
    else
    {
      static metal::Kernel<T>
        kernel("UnaryOperationRank" + std::to_string(rank) + std::string(TOperator<device::Metal>::kernel_name));

      encoder->setBytes(strides_x.data(), strides_x.size() * sizeof(size_t), 3);

      auto [ grid_size, group_size] = GetBlockSize<rank>(dimensions);
      encoder.DispatchThreads(grid_size, group_size);

      device.Wait(); // TODO: use callback or manage dispaltched jobs
    }
  }


 public:
  template<std::ranges::input_range I,
           std::ranges::output_range<std::iter_value_t<std::ranges::iterator_t<I>>> O>
  requires std::indirectly_copyable<std::ranges::iterator_t<I>, std::ranges::iterator_t<O>>
  void operator()(I&& in, O&& out) const
  {
    using value_type = std::iter_value_t<std::ranges::iterator_t<O>>;

    auto first_d = std::ranges::begin(out);
    auto first_x = std::ranges::cbegin(in);

    std::span strides_d(first_d.Strides());
    std::span strides_x(first_x.Strides());

    Fold([&](const auto dimensions, const auto strides_d, const auto strides_x) {
      if (IsContiguous(strides_d, strides_x))
        Eval<value_type>(first_d.Buffer(), first_x.Buffer(),
                         first_d.Offset(), first_x.Offset(),
                         dimensions,
// FIXME was: strides_d.template first<(dimensions.size() > 0) ? dimensions.size() - 1 : 0>(),
                         std::move(strides_d),
                         std::move(strides_x));
      else
        Eval<value_type>(first_d.Buffer(), first_x.Buffer(),
                         first_d.Offset(), first_x.Offset(),
                         dimensions,
                         std::move(strides_d),
                         std::move(strides_x));
    }, first_d.Extents(), first_d.Strides(), first_x.Strides());
  }
};

template <> struct CopyOperator<device::Metal> { static constexpr std::string_view kernel_name = "Copy";  };
template <> struct NegOperator<device::Metal> { static constexpr std::string_view kernel_name = "Neg";  };
template <> struct SiluFunction<device::Metal> { static constexpr std::string_view kernel_name = "Silu";  };

} // end of namespace libai

#endif  // LIBAI_TENSOR_METAL_UNARY_H
