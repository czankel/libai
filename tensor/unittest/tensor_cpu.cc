//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include "gtest/gtest.h"


#include <grid/tensor/tensor.h>
#include <grid/tensor/cpu/queue.h>
#include "tensor_cpu.h"

TEST(CPU, SimpleQueue1)
{
  grid::cpu::Queue queue;
  size_t idx;

  size_t sizes_1[1]{8};
  size_t dims_1[1]{40};
  size_t completed_1[40] = {0};
  // FIXME: use std::s..
  queue.Enqueue(dims_1, sizes_1, [&dims_1, &sizes_1, &completed_1](size_t(&&pos)[1]) -> bool {
      for (size_t c = pos[0] * sizes_1[0]; c < (pos[0] + 1) * sizes_1[0] && c < dims_1[0]; c++)
        completed_1[c] = c;
      return false;
  });
  queue.Sync();

  for (idx = 0; idx < dims_1[0] && completed_1[idx] == idx; idx++)
    ;
  EXPECT_EQ(idx, dims_1[0]);

  // ensure sync can be run again
  queue.Sync();
}

TEST(CPU, SimpleQueue2)
{
  grid::cpu::Queue queue;
  size_t idx;

  size_t sizes_2[2]{3, 8};
  size_t dims_2[2]{32, 20};
  size_t completed_2[32 * 20] = {0};

  queue.Enqueue(dims_2, sizes_2, [&dims_2, &sizes_2, &completed_2](size_t(&&pos)[2]) -> bool {
      for (size_t r = pos[0] * sizes_2[0]; r < (pos[0] + 1) * sizes_2[0] && r < dims_2[0]; r++)
        for (size_t c = pos[1] * sizes_2[1]; c < (pos[1] + 1) * sizes_2[1] && c < dims_2[1]; c++)
          completed_2[r * dims_2[1] + c] = r * dims_2[1] + c;
      return false;
  });
  queue.Sync();

  for (idx = 0; idx < dims_2[0] * dims_2[1] && completed_2[idx] == idx; idx++)
    ;
  EXPECT_EQ(idx, dims_2[0] * dims_2[1]);
}

TEST(CPU, SimpleQueue3)
{
  grid::cpu::Queue queue;
  size_t idx;

  size_t sizes_3[3]{8, 5, 5};
  size_t dims_3[3]{7, 18, 16};
  size_t completed_3[7 * 18 * 16] = {0};

  queue.Enqueue(dims_3, sizes_3, [&dims_3, &sizes_3, &completed_3](size_t(&&pos)[3]) -> bool {
      for (size_t d = pos[0] * sizes_3[0]; d < (pos[0] + 1) * sizes_3[0] && d < dims_3[0]; d++)
        for (size_t r = pos[1] * sizes_3[1]; r < (pos[1] + 1) * sizes_3[1] && r < dims_3[1]; r++)
          for (size_t c = pos[2] * sizes_3[2]; c < (pos[2] + 1) * sizes_3[2] && c < dims_3[2]; c++)
            completed_3[(d * dims_3[1] + r) * dims_3[2] + c] = (d * dims_3[1] + r) * dims_3[2] + c;
      return false;
      });
  queue.Sync();

  for (idx = 0; idx < dims_3[0] * dims_3[1] * dims_3[2] && completed_3[idx] == idx; idx++)
    ;
  EXPECT_EQ(idx, dims_3[0] * dims_3[1] * dims_3[2]);
}
