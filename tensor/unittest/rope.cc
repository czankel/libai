//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/tensor.h>
#include <grid/tensor/function.h>

// FIXME IFDEF??
/// CPU
#include <grid/tensor/base/rope.h>
#include <grid/tensor/base/tensor.h>
#include "tensor_base.h"

/// Metal
#if 0
#include <grid/tensor/metal/rope.h>
#include <grid/tensor/metal/tensor.h>
#include "tensor_metal.h"
#endif

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using testing::ElementsAre;

template <typename T> class RopeTestSuite : public testing::Test {};
TYPED_TEST_SUITE_P(RopeTestSuite);

// use this as a baseline funciton to compare results
template <typename T>
void Rope(T* data, size_t pos, size_t rows, size_t cols)
{
  for (size_t i = 0; i < rows * cols; i+=2)
    {
      float rot = (float) pos / powf(10000.0f, (float)(i % cols) / (float)cols);
      float fcr = cosf(rot);
      float fci = sinf(rot);

      float v0 = data[i];
      float v1 = data[i+1];
      data[i]   = v0 * fcr - v1 * fci;
      data[i+1] = v0 * fci + v1 * fcr;
    }
}


TYPED_TEST_P(RopeTestSuite, TensorRopeTestRank1)
{
  typename TypeParam::Tensor tensor =
    grid::Tensor{ 5.f, 3.f, 1.f, 9.f, 3.f, 2.f, 56.f, 7.f, 1.f, 34.f, 52.f, 65.f, 98.f, 13.f };

  typename TypeParam::Tensor result = Rope(tensor, 1);
  typename TypeParam::Tensor expected = tensor.View();
  Rope(expected.Data(), 1UL, 1UL, tensor.Dimensions()[0]);

  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(RopeTestSuite, TensorRopeTestRank2)
{
  typename TypeParam::Tensor tensor =
    grid::Tensor{ { 5.f,  3.f, 1.f,  9.f,  3.f,  2.f},
                  {56.f, 1.f, 34.f, 52.f, 65.f,  9.f } };

  typename TypeParam::Tensor expected = tensor.View();
  Rope(expected.Data(), 2UL, tensor.Dimensions()[0], tensor.Dimensions()[1]);

  typename TypeParam::Tensor result = Rope(tensor, 2);
  EXPECT_EQ(result, expected);
}

REGISTER_TYPED_TEST_SUITE_P(RopeTestSuite,
    TensorRopeTestRank1,
    TensorRopeTestRank2
);

INSTANTIATE_TYPED_TEST_SUITE_P(RopeTestCPU, RopeTestSuite, TensorCPUType);
// IFDEF METAL
#if 0
INSTANTIATE_TYPED_TEST_SUITE_P(MetalTestCPU, RopeTestSuite, TensorMetalType);
#endif
