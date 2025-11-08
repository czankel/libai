//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_MEMORY_H
#define GRID_TENSOR_MEMORY_H

namespace libai {

/// DeviceMemroy defines a dynamically allocated buffer for the specific device.
template <typename> struct DeviceMemory {};

/// StaticMemory defines constant static memory in the RO section.
template <size_t...> struct StaticMemory {};

/// MemoryMapped defines a memory mapped file.
struct MemoryMapped {};

/// Saclar is a single (read-write) scalar value.
struct Scalar {};

} // end of namespace libai

#endif  // GRID_TENSOR_MEMORY_H
