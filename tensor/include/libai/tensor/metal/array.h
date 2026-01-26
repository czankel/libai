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

#include "../array.h"
#include "../tensor_parameters.h"

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
    auto& device = device::Metal::GetDevice();
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
