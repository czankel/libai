//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include "gtest/gtest.h"


#include <libai/tensor/tensor.h>
#include <libai/tensor/cpu/queue.h>
#include "tensor_cpu.h"

TEST(CPU, SimpleQueue1)
{
  libai::cpu::Queue queue;
  size_t idx;

  std::array<size_t, 1> sizes{8};
  std::array<size_t, 1> dims{40};
  size_t completed[40] = {0};

  queue.Enqueue(dims, sizes, [&completed](auto&& pos, auto dims, auto sizes) -> bool {
      for (size_t c = pos[0] * sizes[0]; c < (pos[0] + 1) * sizes[0] && c < dims[0]; c++)
        completed[c] = c;
      return false;
  });
  queue.Sync();

  for (idx = 0; idx < dims[0] && completed[idx] == idx; idx++)
    ;
  EXPECT_EQ(idx, dims[0]);

  // ensure sync can be run again
  queue.Sync();
}

TEST(CPU, SimpleQueue2)
{
  libai::cpu::Queue queue;
  size_t idx;

  std::array<size_t, 2> sizes{3, 8};
  std::array<size_t, 2> dims{32, 20};
  size_t completed[32 * 20] = {0};

  queue.Enqueue(dims, sizes, [&completed](auto&& pos, auto dims, auto sizes) -> bool {
      for (size_t r = pos[0] * sizes[0]; r < (pos[0] + 1) * sizes[0] && r < dims[0]; r++)
        for (size_t c = pos[1] * sizes[1]; c < (pos[1] + 1) * sizes[1] && c < dims[1]; c++)
          completed[r * dims[1] + c] = r * dims[1] + c;
      return false;
  });
  queue.Sync();

  for (idx = 0; idx < dims[0] * dims[1] && completed[idx] == idx; idx++)
    ;
  EXPECT_EQ(idx, dims[0] * dims[1]);
}

TEST(CPU, SimpleQueue3)
{
  libai::cpu::Queue queue;
  size_t idx;

  size_t sizes[3]{8, 5, 5};
  size_t dims[3]{7, 18, 16};
  size_t completed[7 * 18 * 16] = {0};

  queue.Enqueue(dims, sizes, [&completed](auto&& pos, auto dims, auto sizes) -> bool {
      for (size_t d = pos[0] * sizes[0]; d < (pos[0] + 1) * sizes[0] && d < dims[0]; d++)
        for (size_t r = pos[1] * sizes[1]; r < (pos[1] + 1) * sizes[1] && r < dims[1]; r++)
          for (size_t c = pos[2] * sizes[2]; c < (pos[2] + 1) * sizes[2] && c < dims[2]; c++)
            completed[(d * dims[1] + r) * dims[2] + c] = (d * dims[1] + r) * dims[2] + c;
      return false;
  });
  queue.Sync();

  for (idx = 0; idx < dims[0] * dims[1] * dims[2] && completed[idx] == idx; idx++)
    ;

  EXPECT_EQ(idx, dims[0] * dims[1] * dims[2]);
}
