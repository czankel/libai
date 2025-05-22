//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include "gtest/gtest.h"


#include <grid/tensor/tensor.h>
#include <grid/tensor/base/queue.h>
#include "tensor_base.h"

TEST(Base, SimpleQueue3)
{
  grid::base::Queue queue;

  size_t dims[3]{100, 200, 300};

  queue.Enqueue(dims, [](size_t tile[3]) -> bool {
      printf("dims %lu %lu %lu\n", tile[0], tile[1], tile[2]);
      return false;
      });
  queue.Sync();

  //size_t[2] dims{100};
  //size_t[2] dims{100, 200, 300};
}
