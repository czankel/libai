//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/cpu/device.h>

using namespace grid::device;

grid::device::CPU grid::device::CPU::g_device_;
size_t grid::device::CPU::concurrent_thread_count_ = 0;
