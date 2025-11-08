//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <libai/tensor/tensor.h>
#include <libai/tensor/generator.h>
#include <libai/tensor/precision.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <libai/tensor/cpu/tensor.h>
#include <libai/tensor/cpu/generator.h>
#include <libai/tensor/cpu/softmax.h>
#include "tensor_cpu.h"

#ifdef BUILD_METAL
#include <libai/tensor/metal/tensor.h>
#include <libai/tensor/metal/softmax.h>
#include "tensor_metal.h"
#endif

#ifdef BUILD_CUDA
#include <libai/tensor/cuda/tensor.h>
#include <libai/tensor/cuda/softmax.h>
#include "tensor_cuda.h"
#endif


using testing::ElementsAre;


template <typename T> class SoftMaxTestSuite : public testing::Test {};
TYPED_TEST_SUITE_P(SoftMaxTestSuite);


TYPED_TEST_P(SoftMaxTestSuite, TensorSoftMaxRank1)
{
  typename TypeParam::Tensor tensor =
    libai::Tensor{ 1.618f, 2.f, 3.14f, 5.382f, -8.5f, 1.3f, -2.1f, 3.477f, 5.5f };
  libai::Tensor expected {
    0.944665670912764495e-02f, 0.138413555743161758e-01f, 0.432787127410648800e-01f,
    0.407345162648720671e-00f, 0.381141786703997885e-06f, 0.687341376275671639e-02f,
    0.229388293053500676e-03f, 0.606221835647282256e-01f, 0.458362745564445395e-00f };

  libai::Precision p(100.f);
  typename TypeParam::Tensor result = libai::SoftMax(tensor);
  EXPECT_EQ(result, expected);
}


TYPED_TEST_P(SoftMaxTestSuite, TensorSoftMaxRank2Large)
{
  libai::Precision p(100.f);
  auto random = libai::Random<libai::Tensor, float>({10000,7000})();

  typename TypeParam::Tensor tensor{random};
  typename TypeParam::Tensor result = libai::SoftMax(tensor);

  libai::Tensor expected = libai::SoftMax(random);
  EXPECT_EQ(result, expected);
}


REGISTER_TYPED_TEST_SUITE_P(SoftMaxTestSuite,
    TensorSoftMaxRank1,
    TensorSoftMaxRank2Large);

INSTANTIATE_TYPED_TEST_SUITE_P(SoftMaxTestCPU, SoftMaxTestSuite, TensorCPUType);
#ifdef BUILD_METAL
INSTANTIATE_TYPED_TEST_SUITE_P(SoftMaxTestMetal, SoftMaxTestSuite, TensorMetalType);
#endif
#ifdef BUILD_CUDA
INSTANTIATE_TYPED_TEST_SUITE_P(SoftMaxTestCuda, SoftMaxTestSuite, TensorCudaType);
#endif
