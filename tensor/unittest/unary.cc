//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <libai/tensor/tensor.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <libai/tensor/cpu/tensor.h>
#include <libai/tensor/cpu/unary.h>
#include "tensor_cpu.h"

#ifdef BUILD_METAL
#include <libai/tensor/metal/tensor.h>
#include <libai/tensor/metal/unary.h>
#include "tensor_metal.h"
#endif

#ifdef BUILD_CUDA
#include <libai/tensor/cuda/tensor.h>
#include <libai/tensor/cuda/unary.h>
#include "tensor_cuda.h"
#endif


using testing::ElementsAre;

template <typename T> class UnaryTestSuite : public testing::Test {};
TYPED_TEST_SUITE_P(UnaryTestSuite);


TYPED_TEST_P(UnaryTestSuite, TensorUnaryElementaryRank0)
{
  typename TypeParam::Tensor tensor = libai::Tensor{ 5 };

  typename TypeParam::Tensor copy = Copy(tensor);
  EXPECT_EQ(copy, libai::Tensor{ 5 });

  typename TypeParam::Tensor neg = Neg(tensor);
  EXPECT_EQ(neg, libai::Tensor{ -5 });
}

TYPED_TEST_P(UnaryTestSuite, TensorUnaryElementaryRank1)
{
  typename TypeParam::Tensor tensor = libai::Tensor{ 11, 22, 33, 44, 55, 66 };

  typename TypeParam::Tensor copy = Copy(tensor);
  EXPECT_EQ(copy, (libai::Tensor{ 11, 22, 33, 44, 55, 66 }));

  typename TypeParam::Tensor neg = Neg(tensor);
  EXPECT_EQ(neg, (libai::Tensor{ -11, -22, -33, -44, -55, -66 }));
}

TYPED_TEST_P(UnaryTestSuite, TensorUnaryElementaryRank2)
{
  typename TypeParam::Tensor tensor = libai::Tensor{ {11, 22}, {33, 44}, {55, 66} };

  typename TypeParam::Tensor copy = Copy(tensor);
  EXPECT_EQ(copy, (libai::Tensor{ {11, 22}, {33, 44}, {55, 66} }));

  typename TypeParam::Tensor neg = Neg(tensor);
  EXPECT_EQ(neg, (libai::Tensor{ {-11, -22}, {-33, -44}, {-55, -66} }));
}

TYPED_TEST_P(UnaryTestSuite, TensorUnaryElementaryRank3)
{
  typename TypeParam::Tensor tensor({400, 300, 500}, 2.1f);

  typename TypeParam::Tensor copy = Copy(tensor);
  EXPECT_EQ(copy, (libai::Tensor({400, 300, 500}, 2.1f)));

  typename TypeParam::Tensor neg = Neg(tensor);
  EXPECT_EQ(neg, (libai::Tensor({400, 300, 500}, -2.1f)));
}

REGISTER_TYPED_TEST_SUITE_P(UnaryTestSuite,
    TensorUnaryElementaryRank0,
    TensorUnaryElementaryRank1,
    TensorUnaryElementaryRank2,
    TensorUnaryElementaryRank3);


INSTANTIATE_TYPED_TEST_SUITE_P(UnaryTestCPU, UnaryTestSuite, TensorCPUType);
#ifdef BUILD_METAL
INSTANTIATE_TYPED_TEST_SUITE_P(UnaryTestMetal, UnaryTestSuite, TensorMetalType);
#endif
#ifdef BUILD_CUDA
INSTANTIATE_TYPED_TEST_SUITE_P(UnaryTestCuda, UnaryTestSuite, TensorCudaType);
#endif
