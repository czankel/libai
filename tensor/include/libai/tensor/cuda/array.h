//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef LIBAI_TENSOR_CUDA_ARRAY_H
#define LIBAI_TENSOR_CUDA_ARRAY_H

#include <span>
#include <stdexcept>

#include "device.h"

#include "../array.h"
#include "../tensor_parameters.h"

template<typename> class CudaPointer;

template <typename T>
struct std::iterator_traits<CudaPointer<T>>
{
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using reference = T&;
  using pointer = T*;
  using iterator_category = std::random_access_iterator_tag;
};


namespace libai {

void CudaMallocManaged(void** ptr, size_t size);
void CudaFree(void* ptr);

template <typename T>
class CudaPointer
{
  template <class U> friend class CudaPointer;
  struct void_type {};

 public:
  using value_type = T;

  CudaPointer() {}
  CudaPointer(std::nullptr_t) {}

  template <typename U>
  requires std::is_same_v<T, const U> CudaPointer(CudaPointer<U>& rhs) : ptr_(rhs.ptr_) {}

  CudaPointer(const CudaPointer& rhs) : ptr_(rhs.ptr_) {}
  CudaPointer(CudaPointer&& rhs) : ptr_(rhs.ptr_) { rhs.ptr_ = nullptr; }
  CudaPointer(value_type* ptr) : ptr_(ptr) {}

  CudaPointer& operator=(const CudaPointer& rhs)
  {
    ptr_ = rhs.ptr_;
    return *this;
  }

  CudaPointer operator=(CudaPointer&& rhs)
  {
    ptr_ = rhs.ptr_;
    rhs.ptr_ = nullptr;
    return *this;
  }

  std::conditional<std::is_void<value_type>::value, void_type, value_type>::type&
  operator*() const { return *ptr_; }
  T* operator->() const { return ptr_; }

  auto& operator++()    { ++ptr_; return *this; }
  auto operator++(int)  { auto tmp = this; ptr_++; return tmp; }
  auto& operator--()    { --ptr_; return *this; }
  auto operator--(int)  { auto tmp = this; ptr_--; return tmp; }

  auto operator+(size_t elems) const { return CudaPointer(ptr_ + elems); }
  auto operator+(CudaPointer r) const { std::ptrdiff_t diff = ptr_ + r.ptr_; return diff; }
  auto operator-(size_t elems) const { return CudaPointer(ptr_ - elems); }
  auto operator-(CudaPointer r) const { std::ptrdiff_t diff = ptr_ - r.ptr_; return diff; }

  explicit operator bool() const { return ptr_ != nullptr; }

  friend bool operator==(CudaPointer l, std::nullptr_t)   { return l.ptr_ == nullptr; }
  friend bool operator==(CudaPointer l, CudaPointer r)    { return l.ptr_ == r.ptr_; }
  friend bool operator<(CudaPointer l, CudaPointer r)     { return l.ptr_ > r.ptr_; }
  friend bool operator>(CudaPointer l, CudaPointer r)     { return l.ptr_ < r.ptr_; }

 private:
  value_type* ptr_ = nullptr;
};


template <typename T>
class CudaAllocator
{
 public:
  using value_type = T;
  using pointer = CudaPointer<T>;

  pointer allocate(size_t size)
  {
    void* ptr;
    CudaMallocManaged(&ptr, size * sizeof(value_type));

    return pointer(static_cast<value_type*>(ptr));
  }

  void deallocate(pointer ptr, size_t n)
  {
    CudaFree(&*ptr);
  }
};



/// brief: Array is a specialization for a dynamically allocated buffer.
template <typename T>
class Array<T, DeviceMemory<device::Cuda>>
{
  using value_type = T;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using allocator = CudaAllocator<T>;

 public:
  Array() = default;

  // @brief Allocates a buffer of the provided size.
  Array(size_t size) : size_(size)
  {
    pointer_ = allocator_.allocate(size);
  }

  // @brief Allocates a buffer of the provided size.
  Array(size_t size, std::type_identity<value_type>) : size_(size)
  {
    pointer_ = allocator_.allocate(size);
  }

  // @brief Constructor for a contiguous array with the provided size with initialization.
  Array(size_t size, value_type init) : size_(size)
  {
    pointer_ = allocator_.allocate(size);
    details::initialize_unsafe(Data(), size_, init);
  }

  // @brief Constructor for a non-contiguous array with the provided dimensions and strides.
  template <size_t N>
  Array(const std::array<size_t, N>& dimensions, const std::array<ssize_t, N>& strides)
    : size_(get_array_size(dimensions, strides))
  {
    pointer_ = allocator_.allocate(size_);
  }

  template <size_t N>
  Array(const std::array<size_t, N>& dimensions,
        const std::array<ssize_t, N>& strides,
        std::type_identity<value_type>)
    : size_(get_array_size(dimensions, strides))
  {
    pointer_ = allocator_.allocate(size_);
  }


  // @brief Constructor for a non-contiguous array with the provided dimensions and strides with initialization.
  template <size_t N>
  Array(const std::array<size_t, N>& dimensions, const std::array<ssize_t, N>& strides, value_type init)
    : size_(get_array_size(dimensions, strides))
  {
    pointer_ = allocator_.allocate(size_);
    details::initialize_unsafe(Data(), std::span(dimensions), std::span(strides), init);
  }

  // @brief Copy constructor of contiguous arrays.
  Array(const Array& other) : size_(other.size_)
  {
    pointer_ = allocator_.allocate(size_);
    memcpy(std::to_address(pointer_), std::to_address(other.pointer_), other.size_);
  }

  // @brief Copy constructor with dimensions and strides
  template <size_t N>
  Array(const_pointer pointer,
        const std::array<size_t, N>& dimensions,
        const std::array<ssize_t, N>& strides1,
        const std::array<ssize_t, N>& strides2)
    : size_(get_array_size(dimensions, strides1))
  {
    pointer_ = allocator_.allocate(size_);
    details::copy_unsafe(std::to_address(pointer_), std::to_address(pointer),
                         std::span<const size_t, N>(dimensions.begin(), N),
                         std::span<const ssize_t, N>(strides1.begin(), N),
                         std::span<const ssize_t, N>(strides2.begin(), N));
  }

  // @brief Move constructor.
  Array(Array&& other)
    : size_(other.size_),
      pointer_(std::move(other.pointer_))
  {
    other.pointer_ = nullptr;
  }

  template <size_t N>
  Array(const Array& other,
        const std::array<size_t, N>& dimensions,
        const std::array<ssize_t, N>& strides1,
        const std::array<ssize_t, N>& strides2)
    : size_(get_array_size(dimensions, strides1))
  {
    pointer_ = allocator_.allocate(size_);
    details::copy_unsafe(std::to_address(pointer_), other.Data(),
                         std::span<const size_t, N>(dimensions.begin(), N),
                         std::span<const ssize_t, N>(strides1.begin(), N),
                         std::span<const ssize_t, N>(strides2.begin(), N));
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
        allocator_.deallocate(pointer_, size_);
      pointer_ = allocator_.allocate(size);
      size_ = size;
    }

    return *this;
  }


  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return size_; }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return &*pointer_; }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return &*pointer_; }

 protected:
  allocator           allocator_;
  size_t              size_;
  allocator::pointer  pointer_;
};


} // end of namespace libai

#endif  // LIBAI_TENSOR_CUDA_ARRAY_H
