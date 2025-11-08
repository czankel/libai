//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/tensor.h>
#include <grid/tensor/mmap.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <grid/tensor/cpu/tensor.h>
#include "tensor_cpu.h"

#ifdef BUILD_METAL
#include <grid/tensor/metal/tensor.h>
#include "tensor_metal.h"
#endif

#ifdef BUILD_CUDA
#include <grid/tensor/cuda/tensor.h>
#include "tensor_cuda.h"
#endif


using testing::ElementsAre;

using libai::view::Slice;
using libai::view::Null;
using libai::view::NewAxis;


// Use Google's Type-Parameterized Tests so these tests can be re-used for other device implementations.

template <typename T> class TensorTestSuite : public testing::Test {};
TYPED_TEST_SUITE_P(TensorTestSuite);


TYPED_TEST_P(TensorTestSuite, TensorBraceInitializationRank0Integer)
{
  typename TypeParam::Tensor tensor = libai::Tensor{ 4 };
  EXPECT_EQ(tensor.Rank(), 0);
}

TYPED_TEST_P(TensorTestSuite, TensorBraceInitializationRank1Integer)
{
  libai::Tensor tensor1{ 11, 22, 33, 44, 55, 66 };
  //typename TypeParam::Tensor tensor2 = { 11, 22, 33, 44, 55, 66 };
  typename TypeParam::Tensor tensor3 = libai::Tensor{ 11, 22, 33, 44, 55, 66 };

  EXPECT_TRUE(libai::is_tensor_v<decltype(tensor1)>);
  EXPECT_EQ(tensor1.Rank(), 1);
  EXPECT_THAT(tensor1.Dimensions(), ElementsAre(6));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(1));

  int data[] = { 11, 22, 33, 44, 55, 66 };
  EXPECT_EQ(memcmp(tensor1.Data(), data, sizeof(data)), 0);
  //EXPECT_EQ(tensor1, tensor2);
  EXPECT_EQ(tensor1, tensor3);
}

TYPED_TEST_P(TensorTestSuite, TensorBraceInitializationRank2Integer)
{
  libai::Tensor tensor1{ { 11, 12 }, { 21, 22, 23 }, { 31, 32, 33, 34 } };
  //typename TypeParam::Tensor tensor2 = { { 11, 12 }, { 21, 22, 23 }, { 31, 32, 33, 34 } };
  typename TypeParam::Tensor tensor3 = libai::Tensor{ { 11, 12 }, { 21, 22, 23 }, { 31, 32, 33, 34 } };

  EXPECT_EQ(tensor1.Rank(), 2);
  EXPECT_THAT(tensor1.Dimensions(), ElementsAre(3, 4));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(4, 1));

  const int* data = reinterpret_cast<const int*>(tensor1.Data());
  EXPECT_EQ(data[0], 11);
  EXPECT_EQ(data[4], 21);
  EXPECT_EQ(data[8], 31);
  EXPECT_EQ(data[9], 32);
  //EXPECT_EQ(tensor1, tensor2);
  EXPECT_EQ(tensor1, tensor3);
}

TYPED_TEST_P(TensorTestSuite, TensorBraceInitializationRank3Integer)
{
  typename TypeParam::Tensor tensor1 = libai::Tensor{ { { 111, 112, 113, 114, 115 },
                                                       { 121, 122, 123, 124, 125 },
                                                       { 131, 132, 133, 134, 135 },
                                                       { 141, 142, 143, 144, 145 } },
                                                     { { 211, 212, 213, 214, 215 },
                                                       { 221, 222, 223, 224, 225 },
                                                       { 231, 232, 233, 234, 235 },
                                                       { 241, 242, 243, 244, 245 } },
                                                     { { 311, 312, 313, 314, 315 },
                                                       { 321, 322, 323, 324, 325 },
                                                       { 331, 332, 333, 334, 335 },
                                                       { 341, 342, 343, 344, 345 } } };

  EXPECT_EQ(tensor1.Rank(), 3);
  EXPECT_THAT(tensor1.Dimensions(), ElementsAre(3, 4, 5));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(4*5, 5, 1));

  const int* data = reinterpret_cast<const int*>(tensor1.Data());
  EXPECT_EQ(data[0],  111);
  EXPECT_EQ(data[6],  122);
  EXPECT_EQ(data[12], 133);
  EXPECT_EQ(data[18], 144);
  EXPECT_EQ(data[20], 211);
  EXPECT_EQ(data[59], 345);
}

TYPED_TEST_P(TensorTestSuite, TensorAllocInitializedRank1Double)
{
  typename TypeParam::Tensor tensor1({4}, 1.2f);

  EXPECT_EQ(tensor1.Rank(), 1);
  EXPECT_THAT(tensor1.Dimensions(), ElementsAre(4));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(1));

  float verify[] = { 1.2f, 1.2f, 1.2f, 1.2f };
  EXPECT_EQ(memcmp(tensor1.Data(), verify, sizeof(verify)), 0);
}

TYPED_TEST_P(TensorTestSuite, TensorAllocUninitializedRank1Double)
{
  typename TypeParam::Tensor tensor1({5}, libai::Uninitialized<double>{});
  EXPECT_EQ(tensor1.Rank(), 1);
  EXPECT_THAT(tensor1.Dimensions(), ElementsAre(5));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(1));
}

TYPED_TEST_P(TensorTestSuite, TensorAllocInitializedRank2Char)
{
  typename TypeParam::Tensor tensor1({5, 4}, (char)'3');

  EXPECT_EQ(tensor1.Rank(), 2);
  EXPECT_THAT(tensor1.Dimensions(), ElementsAre(5, 4));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(4, 1));

  char verify[] = { '3', '3', '3', '3', '3',
                    '3', '3', '3', '3', '3',
                    '3', '3', '3', '3', '3',
                    '3', '3', '3', '3', '3' };
  EXPECT_EQ(memcmp(tensor1.Data(), verify, sizeof(verify)), 0);
}

TYPED_TEST_P(TensorTestSuite, TensorAllocUninitializedRank2Double)
{
  typename TypeParam::Tensor tensor1({7, 3}, libai::Uninitialized<int>{});

  EXPECT_EQ(tensor1.Rank(), 2);
  EXPECT_THAT(tensor1.Dimensions(), ElementsAre(7, 3));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(3, 1));
}

TYPED_TEST_P(TensorTestSuite, TensorAllocInitializedRank3Double)
{
  typename TypeParam::Tensor tensor1{{4, 5, 7}, 3.3};

  EXPECT_EQ(tensor1.Rank(), 3);
  EXPECT_THAT(tensor1.Dimensions(), ElementsAre(4, 5, 7));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(7 * 5, 7, 1));
}

TYPED_TEST_P(TensorTestSuite, TensorAllocUninitializedRank3Double)
{
  typename TypeParam::Tensor tensor({3, 2, 1}, libai::Uninitialized<double>{});
  EXPECT_THAT(tensor.Strides(), ElementsAre(2 * 1, 1, 0));
}

TYPED_TEST_P(TensorTestSuite, TensorAllocUninitializedPattedRank3Double)
{
  typename TypeParam::Tensor tensor1({3, 2, 1}, {2 * 2 * 4, 2 * 2, 2}, libai::Uninitialized<double>{});
  EXPECT_EQ(tensor1.Rank(), 3);
  EXPECT_THAT(tensor1.Dimensions(), ElementsAre(3, 2, 1));
  EXPECT_THAT(tensor1.Strides(), ElementsAre(2 * 2 * 4, 2 * 2, 2));
}

TYPED_TEST_P(TensorTestSuite, TensorMMap)
{
  std::FILE* tmpf = std::tmpfile();

  std::array<size_t, 4> ds1 = {2, 4, 4, 1};
  std::fwrite(ds1.data(), sizeof ds1[0], ds1.size(), tmpf);

  std::array<double, 4> row1 = {1.2, 2.3, 3.4, 4.5};
  std::fwrite(row1.data(), sizeof row1[0], row1.size(), tmpf);
  std::fwrite(row1.data(), sizeof row1[0], row1.size(), tmpf);

  std::array<size_t, 4> ds2 = {2, 3, 3, 1};
  std::fwrite(ds2.data(), sizeof ds2[0], ds2.size(), tmpf);

  std::array<double, 3> row2 = {4.3, 3.2, 2.1};
  std::fwrite(row2.data(), sizeof row2[0], row2.size(), tmpf);
  std::fwrite(row2.data(), sizeof row2[0], row2.size(), tmpf);

  size_t file_size = std::ftell(tmpf);
  EXPECT_EQ(file_size, sizeof(size_t) * 8 + sizeof(double) * (8 + 6));

  std::rewind(tmpf);

  int fd = fileno(tmpf);
  auto mmap = std::shared_ptr<libai::MMap>(libai::MMap::MMapFile(fd, file_size));
  close(fd);

  libai::MMapView view(std::move(mmap));
  auto dimensions1 = view.Read<std::array<size_t, 2>>();
  auto strides1 = view.Read<std::array<ssize_t, 2>>();
  auto size1 = libai::get_array_size<double>(dimensions1, strides1);
  double* addr1 = reinterpret_cast<double*>(view.Address());

  // Note: MemoryMapped Tensor
  libai::Tensor tensor1(dimensions1, std::make_tuple(addr1, size1));

  typename TypeParam::Tensor result1 = libai::Tensor{ {1.2, 2.3, 3.4, 4.5}, {1.2, 2.3, 3.4, 4.5} };
  EXPECT_EQ(tensor1, result1);

  view.Seek(size1 * sizeof(size_t));
  auto dimensions2 = view.Read<std::array<size_t, 2>>();
  auto strides2 = view.Read<std::array<ssize_t, 2>>();
  auto size2 = libai::get_array_size<double>(dimensions2, strides2);
  double* addr2 = reinterpret_cast<double*>(view.Address());

  // Note: MemoryMapped Tensor
  libai::Tensor tensor2(dimensions2, std::make_tuple(addr2, size2));

  typename TypeParam::Tensor result2 = libai::Tensor{ {4.3, 3.2, 2.1}, {4.3, 3.2, 2.1} };
  EXPECT_EQ(tensor2, result2);

  std::fclose(tmpf);
}

TYPED_TEST_P(TensorTestSuite, TensorViewBraceInitializationTensor)
{
  typename TypeParam::Tensor tensor1 = libai::Tensor{ { { 111, 112, 113, 114, 115 },
                                                       { 121, 122, 123, 124, 125 },
                                                       { 131, 132, 133, 134, 135 },
                                                       { 141, 142, 143, 144, 145 } },
                                                     { { 211, 212, 213, 214, 215 },
                                                       { 221, 222, 223, 224, 225 },
                                                       { 231, 232, 233, 234, 235 },
                                                       { 241, 242, 243, 244, 245 } },
                                                     { { 311, 312, 313, 314, 315 },
                                                       { 321, 322, 323, 324, 325 },
                                                       { 331, 332, 333, 334, 335 },
                                                       { 341, 342, 343, 344, 345 } } };

  auto view_row = tensor1.View(1, 2, Slice());
  EXPECT_EQ(view_row.Rank(), 1);
  EXPECT_THAT(view_row.Dimensions(), ElementsAre(5));
  EXPECT_THAT(view_row.Strides(), ElementsAre(1));
  libai::Tensor expected{231, 232, 233, 234, 235};
  EXPECT_EQ(view_row, expected);
  EXPECT_EQ(view_row.Size(), expected.Size());
}

TYPED_TEST_P(TensorTestSuite, TensorViewAllocInitializationTensor)
{
  typename TypeParam::Tensor tensor({4, 5}, 1.1f);
  auto data = tensor.Data();

  // tensor[:,1]
  typename TypeParam::Tensor col = libai::Tensor{2.1f, 3.2f, 4.3f, 5.4f};
  tensor.View(Slice(), 1) = col;
  libai::Tensor expected = { { 1.1f, 2.1f, 1.1f, 1.1f, 1.1f},
                            { 1.1f, 3.2f, 1.1f, 1.1f, 1.1f},
                            { 1.1f, 4.3f, 1.1f, 1.1f, 1.1f},
                            { 1.1f, 5.4f, 1.1f, 1.1f, 1.1f} };

  EXPECT_THAT(tensor.Dimensions(), ElementsAre(4, 5));
  EXPECT_THAT(tensor.Strides(), ElementsAre(5 * 1, 1));
  EXPECT_EQ(tensor, expected);

  // tensor[2]-> (5)
  auto view_index = tensor.View(2);
  EXPECT_EQ(view_index.Rank(), 1);
  EXPECT_THAT(view_index.Dimensions(), ElementsAre(5UL));
  EXPECT_THAT(view_index.Strides(), ElementsAre(1));
  EXPECT_EQ(view_index.Size(), 5); // one row, 5 elements
  EXPECT_EQ(view_index.Data(), data + 2 * 5);

  // tensor[2:] -> (2, 5)
  auto view_span = tensor.View(Slice(2));
  EXPECT_EQ(view_span.Rank(), 2);
  EXPECT_THAT(view_span.Dimensions(), ElementsAre(2UL, 5UL));
  EXPECT_THAT(view_span.Strides(), ElementsAre(5, 1));
  EXPECT_EQ(view_span.Size(), 10);  // 2 x 5 matrix
  EXPECT_EQ(view_span.Data(), data + 2 * 5);
}

TYPED_TEST_P(TensorTestSuite, TensorBroadcast)
{
  typename TypeParam::Tensor tensor({4, 5}, 1.1f);

  // tensor[newaxis] -> (1, 5, 4)
  auto view_newaxis_0 = tensor.View(NewAxis);
  EXPECT_EQ(view_newaxis_0.Rank(), 3);
  EXPECT_THAT(view_newaxis_0.Dimensions(), ElementsAre(1UL, 4UL, 5UL));
  EXPECT_THAT(view_newaxis_0.Strides(), ElementsAre(0, 5, 1));

  // tensor[:,newaxis] -> (4, 1, 5)
  auto view_newaxis_1 = tensor.View(Slice(), NewAxis);
  EXPECT_EQ(view_newaxis_1.Rank(), 3);
  EXPECT_THAT(view_newaxis_1.Dimensions(), ElementsAre(4UL, 1UL, 5UL));
  EXPECT_THAT(view_newaxis_1.Strides(), ElementsAre(5, 0, 1));

  // tensor[:,:,newaxis] -> (4, 5, 1)
  auto view_newaxis_2 = tensor.View(Slice(), Slice(), NewAxis);
  EXPECT_EQ(view_newaxis_2.Rank(), 3);
  EXPECT_THAT(view_newaxis_2.Dimensions(), ElementsAre(4UL, 5UL, 1UL));
  EXPECT_THAT(view_newaxis_2.Strides(), ElementsAre(5, 1, 0));

  // tensor[:,:1:0] -> (4, 1)
  auto view_change_to_broadcast = tensor.View(Slice(), Slice(0, 1, 0));
  EXPECT_EQ(view_change_to_broadcast.Rank(), 2);
  EXPECT_THAT(view_change_to_broadcast.Dimensions(), ElementsAre(4UL, 1UL));
  EXPECT_THAT(view_change_to_broadcast.Strides(), ElementsAre(5, 0));
}


REGISTER_TYPED_TEST_SUITE_P(TensorTestSuite,
    TensorBraceInitializationRank0Integer,
    TensorBraceInitializationRank1Integer,
    TensorBraceInitializationRank2Integer,
    TensorBraceInitializationRank3Integer,
    TensorAllocInitializedRank1Double,
    TensorAllocUninitializedRank1Double,
    TensorAllocInitializedRank2Char,
    TensorAllocUninitializedRank2Double,
    TensorAllocInitializedRank3Double,
    TensorAllocUninitializedRank3Double,
    TensorAllocUninitializedPattedRank3Double,
    TensorMMap,
    TensorViewBraceInitializationTensor,
    TensorViewAllocInitializationTensor,
    TensorBroadcast);


INSTANTIATE_TYPED_TEST_SUITE_P(TensorTestCPU, TensorTestSuite, TensorCPUType);
#ifdef BUILD_METAL
INSTANTIATE_TYPED_TEST_SUITE_P(TensorTestMetal, TensorTestSuite, TensorMetalType);
#endif
#ifdef BUILD_CUDA
INSTANTIATE_TYPED_TEST_SUITE_P(TensorTestCuda, TensorTestSuite, TensorCudaType);
#endif
