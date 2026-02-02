//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <sys/types.h>
#include <stdexcept>

#include "libai/tensor/array.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "tensor_cpu.h"

#ifdef BUILD_METAL
#include "tensor_metal.h"
#endif

#ifdef BUILD_CUDA
#include "tensor_cuda.h"
#endif

template <typename T> class ArrayTestSuite : public testing::Test {};
TYPED_TEST_SUITE_P(ArrayTestSuite);

// GCC fails here applying the current CTAD and alias templates; unclear if it's a bug

#ifndef __GNUC__

TYPED_TEST_P(ArrayTestSuite, ArrayConstructorUninitialized)
{
  typename TypeParam::Array array(10UL, std::type_identity<double>{});
  EXPECT_EQ(10, array.Size());
  EXPECT_TRUE((std::is_same<double&, decltype(*array.Data())>::value));
}

TYPED_TEST_P(ArrayTestSuite, ArrayConstructorWithType)
{
  typename TypeParam::Array array(10UL, 1.1F);
  EXPECT_EQ(10, array.Size());
  EXPECT_EQ(1.1F, array.Data()[0]);
  EXPECT_TRUE((std::is_same<float&, decltype(*array.Data())>::value));
}

TYPED_TEST_P(ArrayTestSuite, ArrayCopyConstructor)
{
  typename TypeParam::Array array(100UL, std::type_identity<float>{});
  typename TypeParam::Array copy(array);
  EXPECT_TRUE((std::is_same<decltype(*array.Data()), decltype(*copy.Data())>::value));
}

TYPED_TEST_P(ArrayTestSuite, ArrayMoveConstructor)
{
  typename TypeParam::Array array(100UL, std::type_identity<float>{});
  auto data = array.Data();
  typename TypeParam::Array move(std::move(array));
  EXPECT_TRUE((std::is_same<decltype(*array.Data()), decltype(*move.Data())>::value));
  EXPECT_EQ(data, move.Data());
}
#endif

TYPED_TEST_P(ArrayTestSuite, ArrayConstructorDimStridesWithType)
{
  std::array dims{5UL, 10UL};
  std::array strides{10L, 20L};
  typename TypeParam::Array array(dims, strides, 1.1F);
  EXPECT_EQ(10 * 20, array.Size());
  EXPECT_EQ(1.1F, array.Data()[0]);
  EXPECT_TRUE((std::is_same<float&, decltype(*array.Data())>::value));
}


REGISTER_TYPED_TEST_SUITE_P(ArrayTestSuite,
#ifndef __GNUC__
    ArrayCopyConstructor,
    ArrayMoveConstructor,
    ArrayConstructorUninitialized,
    ArrayConstructorWithType,
#endif
    ArrayConstructorDimStridesWithType);

INSTANTIATE_TYPED_TEST_SUITE_P(ArrayTestCPU, ArrayTestSuite, TensorCPUType);
#ifdef BUILD_METAL
INSTANTIATE_TYPED_TEST_SUITE_P(ArrayTestMetal, ArrayTestSuite, TensorMetalType);
#endif
#ifdef BUILD_CUDA
INSTANTIATE_TYPED_TEST_SUITE_P(ArrayTestCuda, ArrayTestSuite, TensorCudaType);
#endif
