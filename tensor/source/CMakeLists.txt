# Copyright (C) Chris Zankel. All rights reserved.
# This code is subject to U.S. and other copyright laws and
# intellectual property protections.
#
# The contents of this file are confidential and proprietary to Chris Zankel.

grid_add_sources(gridtensor
	tensor.cc
	mmap.cc
	cpu/device.cc
        cpu/queue.cc
)

if (BUILD_METAL)
  add_subdirectory(metal/kernels)

  grid_add_sources(gridtensor
    metal/device.cc
  )
endif()

if (BUILD_CUDA)
  add_subdirectory(cuda)
endif()
