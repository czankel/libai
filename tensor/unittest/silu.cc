//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/tensor.h>
#include <grid/tensor/generator.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <grid/tensor/cpu/tensor.h>
#include <grid/tensor/cpu/generator.h>
#include <grid/tensor/cpu/unary.h>
#include "tensor_cpu.h"

#ifdef BUILD_METAL
#include <grid/tensor/metal/tensor.h>
#include <grid/tensor/metal/unary.h>
#include "tensor_metal.h"
#endif

#ifdef BUILD_CUDA
#include <grid/tensor/cuda/tensor.h>
#include <grid/tensor/cuda/unary.h>
#include "tensor_cuda.h"
#endif

using testing::ElementsAre;

template <typename T> class SiLUTestSuite : public testing::Test {};
TYPED_TEST_SUITE_P(SiLUTestSuite);


TYPED_TEST_P(SiLUTestSuite, TensorSiLURank2)
{
  typename TypeParam::Tensor tensor = libai::Tensor {
    { 0.5756126046f, 0.3409004211f, 0.5048730969f, 0.9210063219f },
    { 0.4421079159f, 0.1490573883f, 0.4715823531f, 0.0599167943f },
    { 0.0909789801f, 0.7678806186f, 0.4295750260f, 0.0766910911f },
    { 0.8656119704f, 0.4321447611f, 0.4913122654f, 0.6204570532f },
    { 0.0656521916f, 0.5640842915f, 0.1560901403f, 0.9024674296f } };

  typename TypeParam::Tensor expected = libai::Tensor {
    { 0.3684250116f, 0.1992253512f, 0.3148407936f, 0.6587470770f },
    { 0.2691381276f, 0.0800729543f, 0.2903806865f, 0.0308556333f },
    { 0.0475573614f, 0.5245102644f, 0.2602245808f, 0.0398152061f },
    { 0.6092452407f, 0.2620464265f, 0.3048177660f, 0.4034971893f },
    { 0.0339032635f, 0.3595456779f, 0.0841237679f, 0.6420661807f } };

  typename TypeParam::Tensor result = libai::Silu(tensor);
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(SiLUTestSuite, TensorSiLURank2Large)
{
  libai::Precision p(100.f);
  auto random = libai::Random<libai::Tensor, float>({10000,7000})();

  typename TypeParam::Tensor tensor{random};
  typename TypeParam::Tensor result = libai::Silu(tensor);
  libai::Tensor expected = libai::Silu(random);
  EXPECT_EQ(result, expected);
}


REGISTER_TYPED_TEST_SUITE_P(SiLUTestSuite,
    TensorSiLURank2,
    TensorSiLURank2Large);


INSTANTIATE_TYPED_TEST_SUITE_P(SiLUTestCPU, SiLUTestSuite, TensorCPUType);
#ifdef BUILD_METAL
INSTANTIATE_TYPED_TEST_SUITE_P(SiLUTestMetal, SiLUTestSuite, TensorMetalType);
#endif
#ifdef BUILD_CUDA
INSTANTIATE_TYPED_TEST_SUITE_P(SiLUTestCuda, SiLUTestSuite, TensorCudaType);
#endif
