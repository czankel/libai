//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef LIBAI_TENSOR_MEMORY_H
#define LIBAI_TENSOR_MEMORY_H

#include "concepts.h"

namespace libai {

/// DeviceMemroy defines a dynamically allocated buffer for the specific device.
template <typename TDevice> struct DeviceMemory { using device_type = TDevice; };

/// StaticMemory defines constant static memory in the RO section.
template <size_t...> struct StaticMemory { using device_type = device::CPU; };

/// View defines a view of a tensor
template <typename TTensor> struct View { using device_type = typename libai::tensor_device_t<TTensor>; };

/// MemoryMapped defines a memory mapped file.
struct MemoryMapped { using device_type = device::CPU; };

/// Saclar is a single (read-write) scalar value.
struct Scalar {};

} // end of namespace libai

#endif  // LIBAI_TENSOR_MEMORY_H
