//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/generator.h>
#include <grid/tensor/tensor.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <grid/tensor/cpu/binary.h>
#include <grid/tensor/cpu/generator.h>
#include <grid/tensor/cpu/tensor.h>
#include "tensor_cpu.h"

#ifdef BUILD_METAL
#include <grid/tensor/metal/binary.h>
#include <grid/tensor/metal/tensor.h>
#include "tensor_metal.h"
#endif

#ifdef BUILD_CUDA
#include <grid/tensor/cuda/binary.h>
#include <grid/tensor/cuda/tensor.h>
#include "tensor_cuda.h"
#endif


using testing::ElementsAre;

template <typename T> class AdditionTestSuite : public testing::Test {};
TYPED_TEST_SUITE_P(AdditionTestSuite);


TYPED_TEST_P(AdditionTestSuite, TensorAddRank0)
{
  typename TypeParam::Tensor tensor1 = libai::Tensor{ 5 };
  typename TypeParam::Tensor tensor2 = libai::Tensor{ 3 };
  auto op = libai::Add(tensor1, tensor2);
  auto result = op();
  EXPECT_EQ(result.Rank(), 0);
  libai::Tensor expected{ 8 };
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(AdditionTestSuite, TensorAddRank1)
{
  typename TypeParam::Tensor tensor1 = libai::Tensor{ 11, 22, 33, 44, 55, 66 };
  typename TypeParam::Tensor tensor2 = libai::Tensor{ 89, 78, 67, 56, 45, 34 };
  int v1[] = { 100, 100, 100, 100, 100, 100 };

  auto op1a = libai::Add(tensor1, tensor2);
  auto res1a = op1a();
  EXPECT_EQ(memcmp(res1a.Data(), v1, sizeof(v1)), 0);

  auto&& op1b = tensor2 + tensor1;
  auto res1b = op1b();
  EXPECT_EQ(memcmp(res1b.Data(), v1, sizeof(v1)), 0);
}

TYPED_TEST_P(AdditionTestSuite, TensorAddRank1Dim1)
{
  typename TypeParam::Tensor tensor1 = libai::Tensor{ 11 };
  typename TypeParam::Tensor tensor2 = libai::Tensor{ 89 };
  int v1[] = { 100 };

  auto op1a = libai::Add(tensor1, tensor2);
  auto res1a = op1a();
  EXPECT_EQ(memcmp(res1a.Data(), v1, sizeof(v1)), 0);

  auto&& op1b = tensor2 + tensor1;
  auto res1b = op1b();
  EXPECT_EQ(memcmp(res1b.Data(), v1, sizeof(v1)), 0);
}

TYPED_TEST_P(AdditionTestSuite, TensorAddRank3)
{
  typename TypeParam::Tensor tensor1({4UL, 3UL, 5UL}, 2.1f);
  typename TypeParam::Tensor tensor2({4UL, 3UL, 5UL}, 1.3f);
  typename TypeParam::Tensor expected({4UL, 3UL, 5UL}, 3.4f);
  typename TypeParam::Tensor result = tensor1 + tensor2;
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(AdditionTestSuite, TensorAddAdd)
{
  typename TypeParam::Tensor tensor1({4, 3}, 2.1f);
  typename TypeParam::Tensor tensor2({4, 3}, 1.3f);
  typename TypeParam::Tensor tensor3({4, 3}, 2.2f);
  typename TypeParam::Tensor expected = libai::Tensor{ { 5.6f, 5.6f, 5.6f },
                                                      { 5.6f, 5.6f, 5.6f },
                                                      { 5.6f, 5.6f, 5.6f },
                                                      { 5.6f, 5.6f, 5.6f } };
  auto&& oper = tensor1 + tensor2 + tensor3;
  auto result = oper();
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(AdditionTestSuite, TensorAddMatVecBroadcast)
{
  typename TypeParam::Tensor tensor1({4UL}, 1.1f);
  typename TypeParam::Tensor tensor2({4UL, 5UL}, 4.4f);
  typename TypeParam::Tensor result = tensor2 + tensor1.View(libai::view::Slice{}, libai::view::NewAxis);
  typename TypeParam::Tensor expected({4UL, 5UL}, 5.5f);
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(AdditionTestSuite, TensorAddBroadcast)
{
  typename TypeParam::Tensor tensor1({5}, 1.1f);
  typename TypeParam::Tensor tensor2({4, 5}, 4.4f);
  typename TypeParam::Tensor result = Add(tensor1, tensor2);
  typename TypeParam::Tensor expected({4UL, 5UL}, 5.5f);
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(AdditionTestSuite, TensorAddRank2ContiguousLarge)
{
  libai::Tensor random1 = libai::Random<libai::Tensor, float>({10000,7000});
  libai::Tensor random2 = libai::Random<libai::Tensor, float>({10000,7000});

  typename TypeParam::Tensor tensor1{random1};
  typename TypeParam::Tensor tensor2{random2};
  typename TypeParam::Tensor result = tensor1 + tensor2;

  libai::Tensor expected = random1 + random2;
  EXPECT_EQ(result, expected);
}


REGISTER_TYPED_TEST_SUITE_P(AdditionTestSuite,
    TensorAddRank0,
    TensorAddRank1,
    TensorAddRank1Dim1,
    TensorAddRank3,
    TensorAddAdd,
    TensorAddMatVecBroadcast,
    TensorAddBroadcast,
    TensorAddRank2ContiguousLarge);


INSTANTIATE_TYPED_TEST_SUITE_P(AdditionTestCPU, AdditionTestSuite, TensorCPUType);
#ifdef BUILD_METAL
INSTANTIATE_TYPED_TEST_SUITE_P(AdditionTestMetal, AdditionTestSuite, TensorMetalType);
#endif
#ifdef BUILD_CUDA
INSTANTIATE_TYPED_TEST_SUITE_P(AdditionTestCuda, AdditionTestSuite, TensorCudaType);
#endif
