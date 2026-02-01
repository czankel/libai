//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef LIBAI_TENSOR_ALLOCATOR_H
#define LIBAI_TENSOR_ALLOCATOR_H

namespace libai {

// The following declarations describe "non-allocators" that can be used in lieu of an allocator

// TODO: this will be removed once the code changes to allocators
/// DeviceMemroy defines a dynamically allocated buffer for the specific device.
template <typename TDevice> struct DeviceMemory {};

/// View defines a view of a tensor
template <size_t NRank, typename TAllocator> struct View {};

/// StaticResource defines constant static memory in the RO section.
template <size_t...> struct StaticResource {};

/// MemoryMapped defines a memory mapped file.
struct MemoryMapped {};

/// Saclar is a single (read-write) scalar value.
struct Scalar {};

} // end of namespace libai

#endif  // LIBAI_TENSOR_ALLOCATOR_H
