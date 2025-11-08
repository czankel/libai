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
#include <grid/tensor/cpu/rms_norm.h>
#include "tensor_cpu.h"

#ifdef BUILD_METAL
#include <grid/tensor/metal/tensor.h>
#include <grid/tensor/metal/rms_norm.h>
#include "tensor_metal.h"
#endif

#ifdef BUILD_CUDA
#include <grid/tensor/cuda/tensor.h>
#include <grid/tensor/cuda/rms_norm.h>
#include "tensor_cuda.h"
#endif


using testing::ElementsAre;

namespace {
template <typename T> constexpr size_t size(size_t count) { return sizeof(T) * count; }
}

template <typename T> class RmsNormTestSuite : public testing::Test {};
TYPED_TEST_SUITE_P(RmsNormTestSuite);


TYPED_TEST_P(RmsNormTestSuite, TensorRmsNormRank1)
{
  typename TypeParam::Tensor tensor = libai::Tensor{ 1.618f, 2.f, 3.14f, 5.382f, -8.5f, 13.f, -21.f, 34.77f, 55.f };
  float scale = 23.47965324677914722429f;
  libai::Tensor expected{ 1.618f / scale,    2.f / scale, 3.14f / scale,
                         5.382f / scale,  -8.5f / scale,  13.f / scale,
                          -21.f / scale, 34.77f / scale,  55.f / scale };

  typename TypeParam::Tensor result = libai::RmsNorm(tensor);
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(RmsNormTestSuite, TensorRmsNormRank2)
{
  typename TypeParam::Tensor tensor = libai::Tensor{ { 1.618f,   2.f,  3.14f, 5.382f, -8.5f },
                                                    {   13.f, -21.f, 34.77f,   55.f, 43.5f } };
  float scale1 = 4.851669788360596;
  float scale2 = 36.70477676391602;
  libai::Tensor expected {
    { 1.618f / scale1,   2.f / scale1,  3.14f / scale1, 5.382f / scale1, -8.5f / scale1 },
    {   13.f / scale2, -21.f / scale2, 34.77f / scale2,   55.f / scale2, 43.5f / scale2 } };

  typename TypeParam::Tensor result = libai::RmsNorm(tensor);
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(RmsNormTestSuite, TensorRmsNormRank2Large)
{
  libai::Precision p(100.f);
  auto random = libai::Random<libai::Tensor, float>({10000,7000})();

  typename TypeParam::Tensor tensor{random};
  typename TypeParam::Tensor result = libai::RmsNorm(tensor);
  libai::Tensor expected = libai::RmsNorm(random);
  EXPECT_EQ(result, expected);
}


REGISTER_TYPED_TEST_SUITE_P(RmsNormTestSuite,
    TensorRmsNormRank1,
    TensorRmsNormRank2,
    TensorRmsNormRank2Large);


INSTANTIATE_TYPED_TEST_SUITE_P(RmsNormTestCPU, RmsNormTestSuite, TensorCPUType);
#ifdef BUILD_METAL
INSTANTIATE_TYPED_TEST_SUITE_P(RmsNormTestMetal, RmsNormTestSuite, TensorMetalType);
#endif
#ifdef BUILD_CUDA
INSTANTIATE_TYPED_TEST_SUITE_P(RmsNormTestCuda, RmsNormTestSuite, TensorCudaType);
#endif
