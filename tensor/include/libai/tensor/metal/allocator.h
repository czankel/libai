//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef LIBAI_TENSOR_METAL_ALLOCATOR_H
#define LIBAI_TENSOR_METAL_ALLOCATOR_H

#include <Metal/Metal.hpp>
#include "device.h"

namespace libai {
template<typename> class MetalPointer;
}

template <typename T>
struct std::iterator_traits<libai::MetalPointer<T>>
{
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using reference = T&;
  using pointer = T*;
  using iterator_category = std::random_access_iterator_tag;
};


namespace libai {

template <typename T>
class MetalPointer
{
  template <class U> friend class MetalPointer;
  struct void_type {};

 public:
  using value_type = T;

  MetalPointer() {}
  MetalPointer(std::nullptr_t) {}

  MetalPointer(MTL::Buffer* buffer)
    : buffer_(buffer),
      pointer_(reinterpret_cast<value_type*>(buffer->contents()))
  {}

  template <typename U>
  requires std::is_same_v<T, const U>
  MetalPointer(MetalPointer<U>& rhs) : buffer_(rhs.buffer_), pointer_(rhs.pointer_) {}

  MetalPointer(const MetalPointer& rhs) : buffer_(rhs.buffer_), pointer_(rhs.pointer_) {}
  MetalPointer(MetalPointer&& rhs) : buffer_(rhs.buffer_), pointer_(rhs.pointer_)
  {
    rhs.buffer_ = nullptr;
    rhs.pointer_ = nullptr;
  }

  MetalPointer& operator=(const MetalPointer& rhs)
  {
    buffer_ = rhs.buffer_;
    pointer_ = rhs.pointer_;
    return *this;
  }

  MetalPointer operator=(MetalPointer&& rhs)
  {
    buffer_ = rhs.buffer_;
    pointer_ = rhs.pointer_;
    rhs.buffer_ = nullptr;
    rhs.pointer_ = nullptr;
    return *this;
  }

  std::conditional<std::is_void<value_type>::value, void_type, value_type>::type&
  operator*() const     { return *pointer_; }
  T* operator->() const { return pointer_; }

  auto& operator++()    { ++pointer_; return *this; }
  auto operator++(int)  { auto tmp = this; pointer_++; return tmp; }
  auto& operator--()    { --pointer_; return *this; }
  auto operator--(int)  { auto tmp = this; pointer_--; return tmp; }

  auto operator+(size_t elems) const { return MetalPointer(pointer_ + elems); }
  auto operator+(MetalPointer r) const { std::ptrdiff_t diff = pointer_ + r.pointer_; return diff; }
  auto operator-(size_t elems) const { return MetalPointer(pointer_ - elems); }
  auto operator-(MetalPointer r) const { std::ptrdiff_t diff = pointer_ - r.pointer_; return diff; }

  explicit operator bool() const { return pointer_ != nullptr; }

  friend bool operator==(MetalPointer l, std::nullptr_t)    { return l.pointer_ == nullptr; }
  friend bool operator==(MetalPointer l, MetalPointer r)    { return l.pointer_ == r.pointer_; }
  friend bool operator<(MetalPointer l, MetalPointer r)     { return l.pointer_ > r.pointer_; }
  friend bool operator>(MetalPointer l, MetalPointer r)     { return l.pointer_ < r.pointer_; }

  MTL::Buffer* Buffer()                                     { return buffer_; }
  const MTL::Buffer* Buffer() const                         { return buffer_; }

 private:
  MTL::Buffer*  buffer_ = nullptr;
  value_type*   pointer_ = nullptr;
};


template <typename T>
class MetalAllocator
{
 public:
  using value_type = T;
  using pointer = MetalPointer<T>;

  pointer allocate(size_t size)
  {
    size *= sizeof(value_type);

    // Align up memory -- TODO only for larger sizes not < page_size?
    if (size > vm_page_size)
      size = vm_page_size * ((size + vm_page_size - 1) / vm_page_size);

    // Allocate new buffer
    size_t mode = MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked;
    auto& device = libai::device::Metal::GetDevice();
    auto* buffer = device.NewBuffer(size, mode);
    if (buffer == nullptr)
      throw std::runtime_error("failed to allocate buffer");
    return pointer(buffer);
  }

  void deallocate(pointer p, size_t n)
  {
    p.Buffer()->release();
  }
};


} // end of namespace libai

#endif  // LIBAI_TENSOR_METAL_ALLOCATOR_H
