//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef LIBAI_TENSOR_METAL_ARRAY_H
#define LIBAI_TENSOR_METAL_ARRAY_H

#include <span>
#include <stdexcept>

#include "device.h"

#include "allocator.h"
#include "../array.h"
#include "../tensor_parameters.h"

namespace libai {

/// brief: Array is a specialization for a dynamically allocated buffer.
template <typename T>
class Array<T, DeviceMemory<device::Metal>>
{
  using value_type = T;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using allocator = MetalAllocator<T>;

 public:
  Array() = default;

  // TODO: size is in bytes not element numbers, revisit...

  // @brief Constructor for a contiguous array with the provided size.
  Array(size_t size) : size_(size), pointer_(allocator_.allocate(size)) {}

  // @brief Constructor for a contiguous array with the provided size.
  Array(size_t size, std::type_identity<value_type>)
    : size_(size), pointer_(allocator_.allocate(size))
  {}

  // @brief Constructor for a contiguous array with the provided size with initialization.
  Array(size_t size, value_type init) : size_(size), pointer_(allocator_.allocate(size))
  {
    details::initialize_unsafe(Data(), size_, init);
  }

  // @brief Constructor for a non-contiguous array with the provided dimensions and strides.
  template <size_t N>
  Array(const std::array<size_t, N>& dimensions, const std::array<ssize_t, N>& strides)
    : size_(get_array_size(dimensions, strides)),
      pointer_(allocator_.allocate(size_))
  {}

  // @brief Constructor for a non-contiguous array with the provided dimensions and strides.
  template <size_t N>
  Array(const std::array<size_t, N>& dimensions,
        const std::array<ssize_t, N>& strides,
        std::type_identity<value_type>)
    : size_(get_array_size(dimensions, strides)),
      pointer_(allocator_.allocate(size_))
  {}


  // @brief Constructor for a non-contiguous array with the provided dimensions and strides with initialization.
  template <size_t N>
  Array(const std::array<size_t, N>& dimensions, const std::array<ssize_t, N>& strides, value_type init)
    : size_(get_array_size(dimensions, strides)),
      pointer_(allocator_.allocate(size_))
  {
    details::initialize_unsafe(Data(), std::span(dimensions), std::span(strides), init);
  }


  // @brief Copy constructor of contiguous arrays.
  // TODO use GPU for copy
  Array(const Array& other)
    : size_(other.size_),
      pointer_(allocator_.allocate(size_))
  {
    memcpy(Data(), other.Data(), other.size_ * sizeof(value_type));
  }

  // @brief Copy constructor with dimensions and strides
  template <size_t N>
  Array(const_pointer data,
        const std::array<size_t, N>& dimensions,
        const std::array<ssize_t, N>& strides1,
        const std::array<ssize_t, N>& strides2)
    : size_(get_array_size(dimensions, strides1)),
      pointer_(allocator_.allocate(size_))
  {
    details::copy_unsafe(Data(), data,
                         std::span<const size_t, N>(dimensions.begin(), N),
                         std::span<const ssize_t, N>(strides1.begin(), N),
                         std::span<const ssize_t, N>(strides2.begin(), N));
  }

  // @brief Move constructor.
  Array(Array&& other) : size_(other.size_), pointer_(std::move(other.pointer_))
  {
    other.pointer_ = nullptr;
  }


  ~Array()
  {
    if (pointer_ != nullptr)
      allocator_.deallocate(pointer_, size_);
  }

  Array& operator=(Array&& other)
  {
    if (pointer_ != nullptr)
      allocator_.deallocate(pointer_, size_);

    size_ = other.size_;
    pointer_ = std::move(other.pointer_);
    other.pointer_ = nullptr;

    return *this;
  }

  Array& operator=(const Array& other) = delete;

  /// Resize resizes the buffer of the Array. This will destroy
  Array& Realloc(size_t size)
  {
    if (size != size_)
    {
      if (pointer_ != nullptr)
        allocator_->deallocate(pointer_, size_);
      pointer_ = allocator_->allocate(size);
      size_ = size;
    }

    return *this;
  }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return size_; }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return std::to_address(pointer_); }

  /// Data returns a const_pointer to the data buffer.
  const_pointer Data() const                              { return std::to_address(pointer_); }

  // Buffer returns the MTL buffer - internal use only
  MTL::Buffer* Buffer()                                    { return pointer_.Buffer(); }

  // Buffer returns the MTL buffer - internal use only
  const MTL::Buffer* Buffer() const                        { return pointer_.Buffer(); }

 protected:
  allocator           allocator_;
  size_t              size_;
  allocator::pointer  pointer_;
};

} // end of namespace libai

#endif  // LIBAI_TENSOR_METAL_ARRAY_H
