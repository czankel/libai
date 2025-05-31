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
#include <grid/tensor/cpu/matmul.h>
#include <grid/tensor/cpu/tensor.h>
#include "tensor_cpu.h"

#ifdef BUILD_METAL
#include <grid/tensor/metal/binary.h>
#include <grid/tensor/metal/matmul.h>
#include <grid/tensor/metal/tensor.h>
#include "tensor_metal.h"
#endif

#ifdef BUILD_CUDA
#include <grid/tensor/cuda/binary.h>
#include <grid/tensor/cuda/matmul.h>
#include <grid/tensor/cuda/tensor.h>
#include "tensor_cuda.h"
#endif


using testing::ElementsAre;

namespace {
template <typename T> constexpr size_t size(size_t count) { return sizeof(T) * count; }
}

template <typename T> class MultiplicationTestSuite : public testing::Test {};
TYPED_TEST_SUITE_P(MultiplicationTestSuite);


TYPED_TEST_P(MultiplicationTestSuite, TensorVecDot)
{
  // dot -> 14+33+65 = 112
  typename TypeParam::Tensor tensor1 = grid::Tensor{  2,   3,   5 };
  typename TypeParam::Tensor tensor2 = grid::Tensor{  7,  11,  13 };

  typename TypeParam::Tensor result = grid::Matmul(tensor1, tensor2);
  EXPECT_EQ(result, grid::Tensor{ 112 });
}

TYPED_TEST_P(MultiplicationTestSuite, TensorMatmul)
{
  typename TypeParam::Tensor tensor1 = grid::Tensor{ { 3, 6, 9, 5 },
                                                     { 2, 8, 4, 7 },
                                                     { 5, 1, 7, 2 },
                                                     { 9, 3, 1, 5 } };
  typename TypeParam::Tensor tensor2 = grid::Tensor{ { 1, 8, 2 },
                                                     { 5, 3, 3 },
                                                     { 7, 4, 5 },
                                                     { 2, 9, 8 } };

  grid::Tensor expected{
    { 3 * 1 + 6 * 5 + 9 * 7 + 5 * 2, 3 * 8 + 6 * 3 + 9 * 4 + 5 * 9, 3 * 2 + 6 * 3 + 9 * 5 + 5 * 8 },
    { 2 * 1 + 8 * 5 + 4 * 7 + 7 * 2, 2 * 8 + 8 * 3 + 4 * 4 + 7 * 9, 2 * 2 + 8 * 3 + 4 * 5 + 7 * 8 },
    { 5 * 1 + 1 * 5 + 7 * 7 + 2 * 2, 5 * 8 + 1 * 3 + 7 * 4 + 2 * 9, 5 * 2 + 1 * 3 + 7 * 5 + 2 * 8 },
    { 9 * 1 + 3 * 5 + 1 * 7 + 5 * 2, 9 * 8 + 3 * 3 + 1 * 4 + 5 * 9, 9 * 2 + 3 * 3 + 1 * 5 + 5 * 8 } };

  typename TypeParam::Tensor result = grid::Matmul(tensor1, tensor2);
  EXPECT_EQ(result, expected);
}

// contiguous operands
TYPED_TEST_P(MultiplicationTestSuite, TensorMatmulContiguous)
{
  typename TypeParam::Tensor tensor1 = grid::Tensor{ { 3, 6, 9, 5 },
                                                     { 2, 8, 4, 7 },
                                                     { 5, 1, 7, 2 },
                                                     { 9, 3, 1, 5 } };
  typename TypeParam::Tensor tensor2 = grid::Tensor{ { 1, 5, 7, 2 },
                                                     { 8, 3, 4, 9 },
                                                     { 2, 3, 5, 8 } }
  .Reshape(std::array{4UL, 3UL}, std::array{1L, 4L});

  grid::Tensor expected{
    { 3 * 1 + 6 * 5 + 9 * 7 + 5 * 2, 3 * 8 + 6 * 3 + 9 * 4 + 5 * 9, 3 * 2 + 6 * 3 + 9 * 5 + 5 * 8 },
    { 2 * 1 + 8 * 5 + 4 * 7 + 7 * 2, 2 * 8 + 8 * 3 + 4 * 4 + 7 * 9, 2 * 2 + 8 * 3 + 4 * 5 + 7 * 8 },
    { 5 * 1 + 1 * 5 + 7 * 7 + 2 * 2, 5 * 8 + 1 * 3 + 7 * 4 + 2 * 9, 5 * 2 + 1 * 3 + 7 * 5 + 2 * 8 },
    { 9 * 1 + 3 * 5 + 1 * 7 + 5 * 2, 9 * 8 + 3 * 3 + 1 * 4 + 5 * 9, 9 * 2 + 3 * 3 + 1 * 5 + 5 * 8 } };

  typename TypeParam::Tensor result = grid::Matmul(tensor1, tensor2);
  EXPECT_EQ(result, expected);
}


// contiguous only in last dimension
TYPED_TEST_P(MultiplicationTestSuite, TensorMatmulSemiContiguous)
{
  typename TypeParam::Tensor tensor1 = grid::Tensor{ { 3, 6, 9, 5 },
                                                     { 0, 0, 0, 0 },
                                                     { 2, 8, 4, 7 },
                                                     { 0, 0, 0, 0 },
                                                     { 5, 1, 7, 2 },
                                                     { 0, 0, 0, 0 },
                                                     { 9, 3, 1, 5 },
                                                     { 0, 0, 0, 0 } };
  typename TypeParam::Tensor tensor2 = grid::Tensor{ { 1, 5, 7, 2 },
                                                     { 0, 0, 0, 0 },
                                                     { 8, 3, 4, 9 },
                                                     { 0, 0, 0, 0 },
                                                     { 2, 3, 5, 8 },
                                                     { 0, 0, 0, 0 } };

  grid::Tensor expected{
    { 3 * 1 + 6 * 5 + 9 * 7 + 5 * 2, 3 * 8 + 6 * 3 + 9 * 4 + 5 * 9, 3 * 2 + 6 * 3 + 9 * 5 + 5 * 8 },
    { 2 * 1 + 8 * 5 + 4 * 7 + 7 * 2, 2 * 8 + 8 * 3 + 4 * 4 + 7 * 9, 2 * 2 + 8 * 3 + 4 * 5 + 7 * 8 },
    { 5 * 1 + 1 * 5 + 7 * 7 + 2 * 2, 5 * 8 + 1 * 3 + 7 * 4 + 2 * 9, 5 * 2 + 1 * 3 + 7 * 5 + 2 * 8 },
    { 9 * 1 + 3 * 5 + 1 * 7 + 5 * 2, 9 * 8 + 3 * 3 + 1 * 4 + 5 * 9, 9 * 2 + 3 * 3 + 1 * 5 + 5 * 8 } };

  typename TypeParam::Tensor result =
    grid::Matmul(tensor1.Reshape(std::array{4UL,4UL},std::array{8L,1L}),
                 tensor2.Reshape(std::array{4UL,3UL},std::array{1L,8L}));
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(MultiplicationTestSuite, TensorMatmulNonContiguous)
{
  typename TypeParam::Tensor tensor1 = grid::Tensor{ { 3, 0, 6, 0, 9, 0, 5, 0 },
                                                     { 0, 0, 0, 0, 0, 0, 0, 0 },
                                                     { 2, 0, 8, 0, 4, 0, 7, 0 },
                                                     { 0, 0, 0, 0, 0, 0, 0, 0 },
                                                     { 5, 0, 1, 0, 7, 0, 2, 0 },
                                                     { 0, 0, 0, 0, 0, 0, 0, 0 },
                                                     { 9, 0, 3, 0, 1, 0, 5, 0 },
                                                     { 0, 0, 0, 0, 0, 0, 0, 0 } };
  typename TypeParam::Tensor tensor2 = grid::Tensor{ { 1, 0, 5, 0, 7, 0, 2, 0 },
                                                     { 0, 0, 0, 0, 0, 0, 0, 0 },
                                                     { 8, 0, 3, 0, 4, 0, 9, 0 },
                                                     { 0, 0, 0, 0, 0, 0, 0, 0 },
                                                     { 2, 0, 3, 0, 5, 0, 8, 0 },
                                                     { 0, 0, 0, 0, 0, 0, 0, 0 } };
  grid::Tensor expected{
    { 3 * 1 + 6 * 5 + 9 * 7 + 5 * 2, 3 * 8 + 6 * 3 + 9 * 4 + 5 * 9, 3 * 2 + 6 * 3 + 9 * 5 + 5 * 8 },
    { 2 * 1 + 8 * 5 + 4 * 7 + 7 * 2, 2 * 8 + 8 * 3 + 4 * 4 + 7 * 9, 2 * 2 + 8 * 3 + 4 * 5 + 7 * 8 },
    { 5 * 1 + 1 * 5 + 7 * 7 + 2 * 2, 5 * 8 + 1 * 3 + 7 * 4 + 2 * 9, 5 * 2 + 1 * 3 + 7 * 5 + 2 * 8 },
    { 9 * 1 + 3 * 5 + 1 * 7 + 5 * 2, 9 * 8 + 3 * 3 + 1 * 4 + 5 * 9, 9 * 2 + 3 * 3 + 1 * 5 + 5 * 8 } };

  typename TypeParam::Tensor result =
    grid::Matmul(tensor1.Reshape(std::array{4UL,4UL},std::array{16L,2L}),
                 tensor2.Reshape(std::array{4UL,3UL},std::array{2L,16L}));

  EXPECT_EQ(result, expected);
}


TYPED_TEST_P(MultiplicationTestSuite, TensorMatVecContiguous)
{
  typename TypeParam::Tensor tensor1 = grid::Tensor{ { 3, 2, 5, 3 },
                                                     { 6, 8, 1, 7 },
                                                     { 9, 4, 7, 2 } };
  typename TypeParam::Tensor tensor2 = grid::Tensor{ 1, 5, 7, 2 };

  grid::Tensor expected{ 3 * 1 + 2 * 5 + 5 * 7 + 3 * 2,   // 54
                         6 * 1 + 8 * 5 + 1 * 7 + 7 * 2,   // 67
                         9 * 1 + 4 * 5 + 7 * 7 + 2 * 2 }; // 82

  typename TypeParam::Tensor result = grid::Matmul(tensor1, tensor2);
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(MultiplicationTestSuite, TensorMatVecSemiContiguous)
{
  typename TypeParam::Tensor tensor1 = grid::Tensor{ { 3, 2, 5, 3 },
                                                     { 0, 0, 0, 0 },
                                                     { 6, 8, 1, 7 },
                                                     { 0, 0, 0, 0 },
                                                     { 9, 4, 7, 2 },
                                                     { 0, 0, 0, 0 } };
  typename TypeParam::Tensor tensor2 = grid::Tensor{ 1, 5, 7, 2 };

  grid::Tensor expected{ 3 * 1 + 2 * 5 + 5 * 7 + 3 * 2,   // 54
                         6 * 1 + 8 * 5 + 1 * 7 + 7 * 2,   // 67
                         9 * 1 + 4 * 5 + 7 * 7 + 2 * 2 }; // 82

  typename TypeParam::Tensor result =
    grid::Matmul(tensor1.Reshape(std::array{3UL,4UL},std::array{8L,1L}), tensor2);

  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(MultiplicationTestSuite, TensorMatVecNonContiguous)
{
  typename TypeParam::Tensor tensor1 = grid::Tensor{ { 3, 2, 5, 3 },
                                                     { 0, 0, 0, 0 },
                                                     { 6, 8, 1, 7 },
                                                     { 0, 0, 0, 0 },
                                                     { 9, 4, 7, 2 },
                                                     { 0, 0, 0, 0 } };
  typename TypeParam::Tensor tensor2 = grid::Tensor{ 1, 0, 5, 0, 7, 0, 2, 0 };

  grid::Tensor expected{ 3 * 1 + 2 * 5 + 5 * 7 + 3 * 2,   // 54
                         6 * 1 + 8 * 5 + 1 * 7 + 7 * 2,   // 67
                         9 * 1 + 4 * 5 + 7 * 7 + 2 * 2 }; // 82

  typename TypeParam::Tensor result =
    grid::Matmul(tensor1.Reshape(std::array{3UL,4UL},std::array{8L,1L}),
                 tensor2.Reshape(std::array{4UL},std::array{2L}));
  EXPECT_EQ(result, expected);
}

// Note: tests un-optimized: add strides for each (dim_m, dim_n)
TYPED_TEST_P(MultiplicationTestSuite, TensorVecMat)
{
  typename TypeParam::Tensor tensor1 = grid::Tensor{ 1, 5, 7, 9 };
  typename TypeParam::Tensor tensor2 = grid::Tensor{ { 3, 2, 5 },
                                                     { 6, 8, 6 },
                                                     { 9, 4, 7 },
                                                     { 3, 7, 2 } };

  grid::Tensor expected{ 1 * 3 + 5 * 6 + 7 * 9 + 9 * 3,   // 123
                         1 * 2 + 5 * 8 + 7 * 4 + 9 * 7,   // 133
                         1 * 5 + 5 * 6 + 7 * 7 + 9 * 2 }; // 102

  typename TypeParam::Tensor result = grid::Matmul(tensor1, tensor2);
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(MultiplicationTestSuite, TensorVecMatContiguous)
{
  typename TypeParam::Tensor tensor1 = grid::Tensor{ 1, 5, 7, 9 };
  typename TypeParam::Tensor tensor2 = grid::Tensor{ { 3, 6, 9, 3 },
                                                     { 2, 8, 4, 7 },
                                                     { 5, 6, 7, 2 } };

  grid::Tensor expected{ 1 * 3 + 5 * 6 + 7 * 9 + 9 * 3,   // 123
                         1 * 2 + 5 * 8 + 7 * 4 + 9 * 7,   // 133
                         1 * 5 + 5 * 6 + 7 * 7 + 9 * 2 }; // 102

  typename TypeParam::Tensor result =
    grid::Matmul(tensor1, tensor2.Reshape(std::array{4UL,3UL},std::array{1L,4L}));

  EXPECT_EQ(result, expected);
}


TYPED_TEST_P(MultiplicationTestSuite, TensorVecMatSemiContiguous)
{
  typename TypeParam::Tensor tensor1 = grid::Tensor{ { 3, 6, 9 },
                                                     { 0, 0, 0 },
                                                     { 2, 8, 4 },
                                                     { 0, 0, 0 },
                                                     { 5, 1, 7 },
                                                     { 0, 0, 0 } };
  typename TypeParam::Tensor tensor2 = grid::Tensor{ 1, 5, 7 };
  grid::Tensor expected{ 3 * 1 + 6 * 5 + 9 * 7,   // 96
                         2 * 1 + 8 * 5 + 4 * 7,   // 70
                         5 * 1 + 1 * 5 + 7 * 7 }; // 59

  typename TypeParam::Tensor result =
    grid::Matmul(tensor2,
                 tensor1.Reshape(std::array{3UL,3UL},std::array{1L,6L}));
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(MultiplicationTestSuite, TensorVecMatNonContiguous)
{
  typename TypeParam::Tensor tensor1 = grid::Tensor{ { 3, 6, 9 },
                                                     { 0, 0, 0 },
                                                     { 2, 8, 4 },
                                                     { 0, 0, 0 },
                                                     { 5, 1, 7 },
                                                     { 0, 0, 0 } };
  typename TypeParam::Tensor tensor2 = grid::Tensor{ 1, 0, 5, 0, 7, 0 };
  grid::Tensor expected{ 3 * 1 + 6 * 5 + 9 * 7,   // 96
                         2 * 1 + 8 * 5 + 4 * 7,   // 70
                         5 * 1 + 1 * 5 + 7 * 7 }; // 59

  typename TypeParam::Tensor result =
    grid::Matmul(tensor2.Reshape(std::array{3UL},std::array{2L}),
                 tensor1.Reshape(std::array{3UL,3UL},std::array{1L,6L}));
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(MultiplicationTestSuite, TensorScaleRight)
{
  typename TypeParam::Tensor tensor1 = grid::Tensor{ { 4.4f, 6.6f, 8.8f }, { 7.7f, 5.5f, 3.3f } };
  typename TypeParam::Tensor tensor2 = grid::Tensor{ 1.0f/11.f };
  grid::Tensor expected{ { 0.4f, 0.6f, 0.8f }, { 0.7f, 0.5f, 0.3f } };

  typename TypeParam::Tensor result = tensor1 * tensor2;
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(MultiplicationTestSuite, TensorScalexLeft)
{
  typename TypeParam::Tensor tensor1 = grid::Tensor{ 1.0f/11.f };
  typename TypeParam::Tensor tensor2 = grid::Tensor{ { 4.4f, 6.6f, 8.8f }, { 7.7f, 5.5f, 3.3f } };
  grid::Tensor expected{ { 0.4f, 0.6f, 0.8f }, { 0.7f, 0.5f, 0.3f } };

  typename TypeParam::Tensor result = tensor1 * tensor2;
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(MultiplicationTestSuite, TensorElemMulRank1)
{
  typename TypeParam::Tensor tensor1 = grid::Tensor{ 3, 6, 9, 2, 8, 4 };
  typename TypeParam::Tensor tensor2 = grid::Tensor{ 1, 8, 5, 3, 7, 4 };
  grid::Tensor expected{ 3,48,45, 6,56,16 };

  typename TypeParam::Tensor result = grid::Mul(tensor1, tensor2);
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(MultiplicationTestSuite, TensorElemMulRank2)
{
  typename TypeParam::Tensor tensor1 = grid::Tensor{ { 3, 6, 9 }, { 2, 8, 4 }};
  typename TypeParam::Tensor tensor2 = grid::Tensor{ { 1, 8, 5 }, { 3, 7, 4 }};
  grid::Tensor expected{ { 3,48,45 }, { 6,56,16 }};

  typename TypeParam::Tensor result = grid::Mul(tensor1, tensor2);
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(MultiplicationTestSuite, TensorElemMulRank2Broadcast)
{
  typename TypeParam::Tensor tensor1 = grid::Tensor{ { 3, 6, 9 }, { 2, 8, 4 }};
  typename TypeParam::Tensor tensor2 = grid::Tensor{ { 1, 8, 5 } };
  grid::Tensor expected{ { 3,48,45 }, { 2,64,20 }};

  typename TypeParam::Tensor result = grid::Mul(tensor1, tensor2);
  EXPECT_EQ(result, expected);
}

TYPED_TEST_P(MultiplicationTestSuite, TensorElemMulRank2ContiguousLarge)
{
  auto random1 = grid::Random<grid::Tensor, float>({10000,7000})();
  auto random2 = grid::Random<grid::Tensor, float>({10000,7000})();

  typename TypeParam::Tensor tensor1{random1};
  typename TypeParam::Tensor tensor2{random2};
  typename TypeParam::Tensor result = grid::Mul(tensor1, tensor2);

  grid::Tensor expected = grid::Mul(random1, random2);
  EXPECT_EQ(result, expected);
}


REGISTER_TYPED_TEST_SUITE_P(MultiplicationTestSuite,
    TensorVecDot,
    TensorMatmul,
    TensorMatmulContiguous,
    TensorMatmulSemiContiguous,
    TensorMatmulNonContiguous,
    TensorMatVecContiguous,
    TensorMatVecSemiContiguous,
    TensorMatVecNonContiguous,
    TensorVecMat,
    TensorVecMatContiguous,
    TensorVecMatSemiContiguous,
    TensorVecMatNonContiguous,
    TensorScaleRight,
    TensorScalexLeft,
    TensorElemMulRank1,
    TensorElemMulRank2,
    TensorElemMulRank2Broadcast,
    TensorElemMulRank2ContiguousLarge);


INSTANTIATE_TYPED_TEST_SUITE_P(MultiplicationTestCPU, MultiplicationTestSuite, TensorCPUType);
#ifdef BUILD_METAL
INSTANTIATE_TYPED_TEST_SUITE_P(MultiplicationTestMetal, MultiplicationTestSuite, TensorMetalType);
#endif
#ifdef BUILD_CUDA
INSTANTIATE_TYPED_TEST_SUITE_P(MultiplicationTestCuda, MultiplicationTestSuite, TensorCudaType);
#endif
