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

using testing::ElementsAre;

using grid::view::Slice;
using grid::view::Null;
using grid::view::NewAxis;


// Use Google's Type-Parameterized Tests so these tests can be re-used for other device implementations.

template <typename T> class TensorParametersTestSuite : public testing::Test {};
TYPED_TEST_SUITE_P(TensorParametersTestSuite);


TYPED_TEST_P(TensorParametersTestSuite, TensorFoldSingleRank0)
{
  {
    bool callback = false;
    constexpr std::array<size_t, 0>  dims{};
    constexpr std::array<ssize_t, 0> strides{};
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre());
        EXPECT_THAT(f_strides, ElementsAre());
        callback = true;
    }, dims, strides);
    EXPECT_TRUE(callback);
  }
}

TYPED_TEST_P(TensorParametersTestSuite, TensorFoldBroadcastSingleRank0)
{
  {
    bool callback = false;
    constexpr std::array<size_t, 0>  dims{};
    constexpr std::array<ssize_t, 0> strides{};
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre());
        EXPECT_THAT(f_strides, ElementsAre());
        callback = true;
    }, dims, strides);
    EXPECT_TRUE(callback);
  }
}

TYPED_TEST_P(TensorParametersTestSuite, TensorFoldSingleRank1)
{
  // dimension = 1
  {
    bool callback = false;
    constexpr size_t dims[] =     { 1 };
    constexpr ssize_t strides[] = { 2 };
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre());
        EXPECT_THAT(f_strides, ElementsAre());
        callback = true;
    }, std::to_array(dims), std::to_array(strides));
    EXPECT_TRUE(callback);
  }

  // dimension > 1
  {
    bool callback = false;
    constexpr size_t dims[] =     { 3 };
    constexpr ssize_t strides[] = { 2 };
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre(3));
        EXPECT_THAT(f_strides, ElementsAre(2));
        callback = true;
    }, std::to_array(dims), std::to_array(strides));
    EXPECT_TRUE(callback);
  }
}

TYPED_TEST_P(TensorParametersTestSuite, TensorFoldBroadcastSingleRank1)
{
  // dimension = 1
  {
    bool callback = false;
    constexpr size_t dims[] =     { 1 };
    constexpr ssize_t strides[] = { 2 };
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre());
        EXPECT_THAT(f_strides, ElementsAre());
        callback = true;
    }, std::to_array(dims), std::to_array(strides));
    EXPECT_TRUE(callback);
  }

  // dimension > 1
  {
    bool callback = false;
    constexpr size_t dims[] =     { 3 };
    constexpr ssize_t strides[] = { 2 };
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre(3));
        EXPECT_THAT(f_strides, ElementsAre(2));
        callback = true;
    }, std::to_array(dims), std::to_array(strides));
    EXPECT_TRUE(callback);
  }
}


TYPED_TEST_P(TensorParametersTestSuite, TensorFoldSingleRank2)
{
  // foldable
  {
    bool callback = false;
    constexpr size_t dims[] =     {  3, 5 };
    constexpr ssize_t strides[] = { 30, 6 };
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre(15));
        EXPECT_THAT(f_strides, ElementsAre(6));
        callback = true;
    }, std::to_array(dims), std::to_array(strides));
    EXPECT_TRUE(callback);
  }

  // simple "broadcast", foldable
  {
    bool callback = false;
    constexpr size_t dims[] =     { 3, 1 };
    constexpr ssize_t strides[] = { 1, 2 };
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre(3));
        EXPECT_THAT(f_strides, ElementsAre(1));
        callback = true;
    }, std::to_array(dims), std::to_array(strides));
    EXPECT_TRUE(callback);
  }

  // top/bottom "broadcast", foldable
  {
    bool callback = false;
    constexpr size_t dims[] =     { 1, 3, 1 };
    constexpr ssize_t strides[] = { 6, 1, 2 };
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre(3));
        EXPECT_THAT(f_strides, ElementsAre(1));
        callback = true;
    }, std::to_array(dims), std::to_array(strides));
    EXPECT_TRUE(callback);
  }

  // bottom "broadcast", non-contiguous
  {
    bool callback = false;
    constexpr size_t dims[] =     { 3, 1 };
    constexpr ssize_t strides[] = { 3, 2 };
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre(3));
        EXPECT_THAT(f_strides, ElementsAre(3));
        callback = true;
    }, std::to_array(dims), std::to_array(strides));
    EXPECT_TRUE(callback);
  }
}

TYPED_TEST_P(TensorParametersTestSuite, TensorFoldBroadcastSingleRank2)
{
  // foldable
  {
    bool callback = false;
    constexpr size_t dims[] =     {  3, 5 };
    constexpr ssize_t strides[] = { 30, 6 };
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre(15));
        EXPECT_THAT(f_strides, ElementsAre(6));
        callback = true;
    }, std::to_array(dims), std::to_array(strides));
    EXPECT_TRUE(callback);
  }

  // simple "broadcast", foldable
  {
    bool callback = false;
    constexpr size_t dims[] =     { 3, 1 };
    constexpr ssize_t strides[] = { 1, 2 };
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre(3));
        EXPECT_THAT(f_strides, ElementsAre(1));
        callback = true;
    }, std::to_array(dims), std::to_array(strides));
    EXPECT_TRUE(callback);
  }

  // top/bottom "broadcast", foldable
  {
    bool callback = false;
    constexpr size_t dims[] =     { 1, 3, 1 };
    constexpr ssize_t strides[] = { 6, 1, 2 };
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre(3));
        EXPECT_THAT(f_strides, ElementsAre(1));
        callback = true;
    }, std::to_array(dims), std::to_array(strides));
    EXPECT_TRUE(callback);
  }

  // bottom "broadcast", non-contiguous
  {
    bool callback = false;
    constexpr size_t dims[] =     { 3, 1 };
    constexpr ssize_t strides[] = { 3, 2 };
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre(3));
        EXPECT_THAT(f_strides, ElementsAre(3));
        callback = true;
    }, std::to_array(dims), std::to_array(strides));
    EXPECT_TRUE(callback);
  }
}

TYPED_TEST_P(TensorParametersTestSuite, TensorFoldSingleRank3)
{
  // middle "broadcast"
  {
    bool callback = false;
    constexpr size_t dims[] =     { 3, 1, 5 };
    constexpr ssize_t strides[] = { 5, 2, 1 };
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre(15));
        EXPECT_THAT(f_strides, ElementsAre(1));
        callback = true;
    }, std::to_array(dims), std::to_array(strides));
    EXPECT_TRUE(callback);
  }

  // Same as above, but stride prohibits full fold
  {
    bool callback = false;
    constexpr size_t dims[] =     { 3, 1, 5 };
    constexpr ssize_t strides[] = { 6, 2, 1 };
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre(3,5));
        EXPECT_THAT(f_strides, ElementsAre(6,1));
        callback = true;
    }, std::to_array(dims), std::to_array(strides));
    EXPECT_TRUE(callback);
  }

  // Same as above, but longer "skip" or dimensions
  {
    bool callback = false;
    constexpr size_t dims[] =     { 3, 1, 1, 1, 5 };
    constexpr ssize_t strides[] = { 6, 2, 5, 3, 1 };
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre(3,5));
        EXPECT_THAT(f_strides, ElementsAre(6,1));
        callback = true;
    }, std::to_array(dims), std::to_array(strides));
    EXPECT_TRUE(callback);
  }

  // scalar
  {
    bool callback = false;
    constexpr size_t dims[] =     { 4, 5, 6 };
    constexpr ssize_t strides[] = { 0, 0 };
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre(120));
        EXPECT_THAT(f_strides, ElementsAre());
        callback = true;
    }, std::to_array(dims), std::to_array(strides));
    EXPECT_TRUE(callback);
  }

  // non-scalar
  {
    bool callback = false;
    constexpr size_t dims[] =     { 4, 5, 6 };
    constexpr ssize_t strides[] = { 2, 0 };
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre(4, 5, 6));
        EXPECT_THAT(f_strides, ElementsAre(0, 2, 0));
        callback = true;
    }, std::to_array(dims), std::to_array(strides));
    EXPECT_TRUE(callback);
  }
}

TYPED_TEST_P(TensorParametersTestSuite, TensorFoldBroadcastSingleRank3)
{
  // middle "broadcast"
  {
    bool callback = false;
    constexpr size_t dims[] =     { 3, 1, 5 };
    constexpr ssize_t strides[] = { 5, 2, 1 };
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre(15));
        EXPECT_THAT(f_strides, ElementsAre(1));
        callback = true;
    }, std::to_array(dims), std::to_array(strides));
    EXPECT_TRUE(callback);
  }

  // Same as above, but stride prohibits full fold
  {
    bool callback = false;
    constexpr size_t dims[] =     { 3, 1, 5 };
    constexpr ssize_t strides[] = { 6, 2, 1 };
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre(3,5));
        EXPECT_THAT(f_strides, ElementsAre(6,1));
        callback = true;
    }, std::to_array(dims), std::to_array(strides));
    EXPECT_TRUE(callback);
  }

  // fold and expand strides
  {
    bool callback = false;
    constexpr size_t dims[] =     { 3, 5, 1 };
    constexpr ssize_t strides[] =    { 2, 5 };
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre(3,5));
        EXPECT_THAT(f_strides, ElementsAre(0,2));
        callback = true;
    }, std::to_array(dims), std::to_array(strides));
    EXPECT_TRUE(callback);
  }


  // scalar
  {
    bool callback = false;
    constexpr size_t dims[] =     { 4, 5, 6 };
    constexpr ssize_t strides[] =    { 0, 0 };
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre(120));
        EXPECT_THAT(f_strides, ElementsAre());
        callback = true;
    }, std::to_array(dims), std::to_array(strides));
    EXPECT_TRUE(callback);
  }

  // non-scalar
  {
    bool callback = false;
    constexpr size_t dims[] =     { 4, 5, 6 };
    constexpr ssize_t strides[] = { 2, 0 };
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre(4, 5, 6));
        EXPECT_THAT(f_strides, ElementsAre(0, 2, 0));
        callback = true;
    }, std::to_array(dims), std::to_array(strides));
    EXPECT_TRUE(callback);
  }
}

TYPED_TEST_P(TensorParametersTestSuite, TensorFoldOperations)
{
  // dimension > 1 and scalar
  {
    bool callback = false;
    constexpr size_t dims[] =     { 3 };
    constexpr ssize_t strides1[] = { 2 };
    std::array<const ssize_t, 0> strides2{};
    grid::Fold([&callback](auto f_dims, auto f_strides1, auto f_strides2) {
        EXPECT_THAT(f_dims, ElementsAre(3));
        EXPECT_THAT(f_strides1, ElementsAre(2));
        EXPECT_THAT(f_strides2, ElementsAre());
        callback = true;
    }, std::to_array(dims), std::to_array(strides1), strides2);
    EXPECT_TRUE(callback);
  }

  // vector op matrix
  {
    bool callback = false;
    constexpr size_t dims[] =      {  3, 5 };
    constexpr ssize_t strides1[] = {     6 };
    constexpr ssize_t strides2[] = { 30, 6 };
    grid::Fold([&callback](auto f_dims, auto f_strides1, auto f_strides2) {
        EXPECT_THAT(f_dims, ElementsAre(3, 5));
        EXPECT_THAT(f_strides1, ElementsAre(0, 6));
        EXPECT_THAT(f_strides2, ElementsAre(30, 6));
        callback = true;
    }, std::to_array(dims), std::to_array(strides1), std::to_array(strides2));
    EXPECT_TRUE(callback);
  }

  // matrix op vector
  {
    bool callback = false;
    constexpr size_t dims[] =      {  3, 5 };
    constexpr ssize_t strides1[] = { 30, 6 };
    constexpr ssize_t strides2[] = {     6 };
    grid::Fold([&callback](auto f_dims, auto f_strides1, auto f_strides2) {
        EXPECT_THAT(f_dims, ElementsAre(3, 5));
        EXPECT_THAT(f_strides1, ElementsAre(30, 6));
        EXPECT_THAT(f_strides2, ElementsAre(0, 6));
        callback = true;
    }, std::to_array(dims), std::to_array(strides1), std::to_array(strides2));
    EXPECT_TRUE(callback);
  }

  // scalar and tensor
  {
    bool callback = false;
    constexpr size_t dims[] =     {  4, 5, 6 };
    constexpr ssize_t strides1[] = {     0, 0 };
    constexpr ssize_t strides2[] = { 30, 6, 1 };
    grid::Fold([&callback](auto f_dims, auto f_strides1, auto f_strides2) {
        EXPECT_THAT(f_dims, ElementsAre(120));
        EXPECT_THAT(f_strides1, ElementsAre());
        EXPECT_THAT(f_strides2, ElementsAre(1));
        callback = true;
    }, std::to_array(dims), std::to_array(strides1), std::to_array(strides2));
    EXPECT_TRUE(callback);
  }

  // two tensors, strides1 lower-rank -> discontiguous
  {
    bool callback = false;
    constexpr size_t dims[] =      {  4, 5, 6 };
    constexpr ssize_t strides1[] = {     6, 1 };
    constexpr ssize_t strides2[] = { 30, 6, 1 };
    grid::Fold([&callback](auto f_dims, auto f_strides1, auto f_strides2) {
        EXPECT_THAT(f_dims, ElementsAre(4,30));
        EXPECT_THAT(f_strides1, ElementsAre(0, 1));
        EXPECT_THAT(f_strides2, ElementsAre(30,1));
        callback = true;
    }, std::to_array(dims), std::to_array(strides1), std::to_array(strides2));
    EXPECT_TRUE(callback);
  }

  // two tensors
  // each vector element of tensor{1,0,0} is applied to each matrix tensor{5,6}
  // the lower two ranks (matrix) are contiguous
  {
    bool callback = false;
    constexpr size_t dims[] =     {  4, 5, 6 };
    constexpr ssize_t strides1[] = {  1, 0, 0 };
    constexpr ssize_t strides2[] = { 30, 6, 1 };
    grid::Fold([&callback](auto f_dims, auto f_strides1, auto f_strides2) {
        EXPECT_THAT(f_dims, ElementsAre(4,30));
        EXPECT_THAT(f_strides1, ElementsAre(1,0));
        EXPECT_THAT(f_strides2, ElementsAre(30,1));
        callback = true;
    }, std::to_array(dims), std::to_array(strides1), std::to_array(strides2));
    EXPECT_TRUE(callback);
  }

  // two tensors as above, the top stride doesn't change it
  {
    bool callback = false;
    constexpr size_t dims[] =     {  4, 5, 6 };
    constexpr ssize_t strides1[] = {  2, 0, 0 };
    constexpr ssize_t strides2[] = { 30, 6, 1 };
    grid::Fold([&callback](auto f_dims, auto f_strides1, auto f_strides2) {
        EXPECT_THAT(f_dims, ElementsAre(4,30));
        EXPECT_THAT(f_strides1, ElementsAre(2,0));
        EXPECT_THAT(f_strides2, ElementsAre(30,1));
        callback = true;
    }, std::to_array(dims), std::to_array(strides1), std::to_array(strides2));
    EXPECT_TRUE(callback);
  }

  // two tensors, second tensor non-contiguous
  // the stride of the matrix is not contigous
  {
    bool callback = false;
    constexpr size_t dims[] =     {  4, 5, 6 };
    constexpr ssize_t strides1[] = {  2, 0, 0 };
    constexpr ssize_t strides2[] = { 30, 6, 2 };
    grid::Fold([&callback](auto f_dims, auto f_strides1, auto f_strides2) {
        EXPECT_THAT(f_dims, ElementsAre(4, 5, 6));
        EXPECT_THAT(f_strides1, ElementsAre(2,0,0));
        EXPECT_THAT(f_strides2, ElementsAre(30,6,2));
        callback = true;
    }, std::to_array(dims), std::to_array(strides1), std::to_array(strides2));
    EXPECT_TRUE(callback);
  }

  // two tensors with strides as above, but full "broadcast" -> two scalars
  {
    bool callback = false;
    constexpr size_t dims[] =     {  1, 1, 1 };
    constexpr ssize_t strides1[] = {  2, 0, 0 };
    constexpr ssize_t strides2[] = { 30, 6, 2 };
    grid::Fold([&callback](auto f_dims, auto f_strides1, auto f_strides2) {
        EXPECT_THAT(f_dims, ElementsAre());
        EXPECT_THAT(f_strides1, ElementsAre());
        EXPECT_THAT(f_strides2, ElementsAre());
        callback = true;
    }, std::to_array(dims), std::to_array(strides1), std::to_array(strides2));
    EXPECT_TRUE(callback);
  }
}

TYPED_TEST_P(TensorParametersTestSuite, TensorFoldBroadcastOperations)
{
  // dimension > 1 and scalar
  {
    bool callback = false;
    constexpr size_t dims[] =     { 3 };
    constexpr ssize_t strides1[] = { 2 };
    std::array<const ssize_t, 0> strides2{};
    grid::Fold([&callback](auto f_dims, auto f_strides1, auto f_strides2) {
        EXPECT_THAT(f_dims, ElementsAre(3));
        EXPECT_THAT(f_strides1, ElementsAre(2));
        EXPECT_TRUE(f_strides2.empty());
        callback = true;
    }, std::to_array(dims), std::to_array(strides1), strides2);
    EXPECT_TRUE(callback);
  }

  // vector and matrix
  {
    bool callback = false;
    constexpr size_t dims[] =      {  3, 5 };
    constexpr ssize_t strides1[] = {     6 };
    constexpr ssize_t strides2[] = { 30, 6 };
    grid::Fold([&callback](auto f_dims, auto f_strides1, auto f_strides2) {
        EXPECT_THAT(f_dims, ElementsAre(3, 5));
        EXPECT_THAT(f_strides1, ElementsAre(0, 6));
        EXPECT_THAT(f_strides2, ElementsAre(30, 6));
        callback = true;
    }, std::to_array(dims), std::to_array(strides1), std::to_array(strides2));
    EXPECT_TRUE(callback);
  }

  // matrix and vector
  {
    bool callback = false;
    constexpr size_t dims[] =      {  3, 5 };
    constexpr ssize_t strides1[] = { 30, 6 };
    constexpr ssize_t strides2[] = {     6 };
    grid::Fold([&callback](auto f_dims, auto f_strides1, auto f_strides2) {
        EXPECT_THAT(f_dims, ElementsAre(3, 5));
        EXPECT_THAT(f_strides1, ElementsAre(30, 6));
        EXPECT_THAT(f_strides2, ElementsAre(0, 6));
        callback = true;
    }, std::to_array(dims), std::to_array(strides1), std::to_array(strides2));
    EXPECT_TRUE(callback);
  }

  // scalar and tensor
  {
    bool callback = false;
    constexpr size_t dims[] =     {  4, 5, 6 };
    constexpr ssize_t strides1[] = {     0, 0 };
    constexpr ssize_t strides2[] = { 30, 6, 1 };
    grid::Fold([&callback](auto f_dims, auto f_strides1, auto f_strides2) {
        EXPECT_THAT(f_dims, ElementsAre(120));
        EXPECT_THAT(f_strides1, ElementsAre());
        EXPECT_THAT(f_strides2, ElementsAre(1));
        callback = true;
    }, std::to_array(dims), std::to_array(strides1), std::to_array(strides2));
    EXPECT_TRUE(callback);
  }

  // two tensors, strides1 lower-rank -> discontiguous
  {
    bool callback = false;
    constexpr size_t dims[] =      {  4, 5, 6 };
    constexpr ssize_t strides1[] = {     6, 1 };
    constexpr ssize_t strides2[] = { 30, 6, 1 };
    grid::Fold([&callback](auto f_dims, auto f_strides1, auto f_strides2) {
        EXPECT_THAT(f_dims, ElementsAre(4,30));
        EXPECT_THAT(f_strides1, ElementsAre(0,1));
        EXPECT_THAT(f_strides2, ElementsAre(30,1));
        callback = true;
    }, std::to_array(dims), std::to_array(strides1), std::to_array(strides2));
    EXPECT_TRUE(callback);
  }

  // two tensors
  // each vector element of tensor{1,0,0} is applied to each matrix tensor{5,6}
  // the lower two ranks (matrix) are contiguous
  {
    bool callback = false;
    constexpr size_t dims[] =     {  4, 5, 6 };
    constexpr ssize_t strides1[] = {  1, 0, 0 };
    constexpr ssize_t strides2[] = { 30, 6, 1 };
    grid::Fold([&callback](auto f_dims, auto f_strides1, auto f_strides2) {
        EXPECT_THAT(f_dims, ElementsAre(4,30));
        EXPECT_THAT(f_strides1, ElementsAre(1,0));
        EXPECT_THAT(f_strides2, ElementsAre(30,1));
        callback = true;
    }, std::to_array(dims), std::to_array(strides1), std::to_array(strides2));
    EXPECT_TRUE(callback);
  }

  // two tensors as above, the top stride doesn't change it
  {
    bool callback = false;
    constexpr size_t dims[] =     {  4, 5, 6 };
    constexpr ssize_t strides1[] = {  2, 0, 0 };
    constexpr ssize_t strides2[] = { 30, 6, 1 };
    grid::Fold([&callback](auto f_dims, auto f_strides1, auto f_strides2) {
        EXPECT_THAT(f_dims, ElementsAre(4,30));
        EXPECT_THAT(f_strides1, ElementsAre(2,0));
        EXPECT_THAT(f_strides2, ElementsAre(30,1));
        callback = true;
    }, std::to_array(dims), std::to_array(strides1), std::to_array(strides2));
    EXPECT_TRUE(callback);
  }

  // two tensors, second tensor non-contiguous
  // the stride of the matrix is not contigous
  {
    bool callback = false;
    constexpr size_t dims[] =     {  4, 5, 6 };
    constexpr ssize_t strides1[] = {  2, 0, 0 };
    constexpr ssize_t strides2[] = { 30, 6, 2 };
    grid::Fold([&callback](auto f_dims, auto f_strides1, auto f_strides2) {
        EXPECT_THAT(f_dims, ElementsAre(4, 5, 6));
        EXPECT_THAT(f_strides1, ElementsAre(2,0,0));
        EXPECT_THAT(f_strides2, ElementsAre(30,6,2));
        callback = true;
    }, std::to_array(dims), std::to_array(strides1), std::to_array(strides2));
    EXPECT_TRUE(callback);
  }

  // two tensors with strides as above, but full "broadcast" -> two scalars
  {
    bool callback = false;
    constexpr size_t dims[] =     {  1, 1, 1 };
    constexpr ssize_t strides1[] = {  2, 0, 0 };
    constexpr ssize_t strides2[] = { 30, 6, 2 };
    grid::Fold([&callback](auto f_dims, auto f_strides1, auto f_strides2) {
        EXPECT_THAT(f_dims, ElementsAre());
        EXPECT_THAT(f_strides1, ElementsAre());
        EXPECT_THAT(f_strides2, ElementsAre());
        callback = true;
    }, std::to_array(dims), std::to_array(strides1), std::to_array(strides2));
    EXPECT_TRUE(callback);
  }
}


  // TODO: top-broadcast only folded if all is foldable, would require finesse with strides...
#if 0
  // Rank 5, top/bottom "broadcast", non-fully-foldable
  {
    bool callback = false;
    constexpr size_t dims[] =     { 1, 1, 4, 3, 1 };
    constexpr ssize_t strides[] = { 12, 2, 2 };
    grid::Fold([&callback](auto f_dims, auto f_strides) {
        EXPECT_THAT(f_dims, ElementsAre(4, 3));
        EXPECT_THAT(f_strides, ElementsAre());
        callback = true;
    }, std::to_array(dims), std::to_array(strides));
    EXPECT_TRUE(callback);
  }
#endif

REGISTER_TYPED_TEST_SUITE_P(TensorParametersTestSuite,
    TensorFoldSingleRank0,
    TensorFoldBroadcastSingleRank0,
    TensorFoldSingleRank1,
    TensorFoldBroadcastSingleRank1,
    TensorFoldSingleRank2,
    TensorFoldBroadcastSingleRank2,
    TensorFoldSingleRank3,
    TensorFoldBroadcastSingleRank3,
    TensorFoldOperations,
    TensorFoldBroadcastOperations
    );

INSTANTIATE_TYPED_TEST_SUITE_P(TensorTestCPU, TensorParametersTestSuite, TensorCPUType);
