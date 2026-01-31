//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef LIBAI_TENSOR_ARRAY_H
#define LIBAI_TENSOR_ARRAY_H

#include <cstring>
#include <span>

#include "allocator.h"
#include "concepts.h"
#include "tensor_parameters.h"

namespace libai {

namespace details {

// TODO: provide optimization with memcpy with contiguous arrays

// copy copies the data between buffers accordig to dimensions and strides.
template <Arithmetic T, Arithmetic S>
inline void copy_unsafe(T* dst, const S* src,
                        std::span<const size_t,  0>,
                        std::span<const ssize_t, 0>,
                        std::span<const ssize_t, 0>)
{
  *dst = *src;
}

template <Arithmetic T, Arithmetic S>
inline void copy_unsafe(T* dst, const S* src,
                        std::span<const size_t,  1> dimensions,
                        std::span<const ssize_t, 1> strides1,
                        std::span<const ssize_t, 1> strides2)
{
  for (size_t i = 0; i < dimensions[0]; i++)
  {
    *dst = *src;
    dst += strides1[0];
    src += strides2[0];
  }
}

template <Arithmetic T, Arithmetic S, size_t N>
inline std::enable_if_t<(N > 1), void>
copy_unsafe(T* dst, const S* src,
            std::span<const size_t,  N> dimensions,
            std::span<const ssize_t, N> strides1,
            std::span<const ssize_t, N> strides2)
{
  static_assert(N != std::dynamic_extent, "dynamic_extent not allowed");
  for (size_t i = 0; i < dimensions[0]; i++)
  {
    copy_unsafe(dst, src,
                std::span<const size_t,  N - 1>(dimensions.begin() + 1, N - 1),
                std::span<const ssize_t, N - 1>(strides1.begin() + 1, N - 1),
                std::span<const ssize_t, N - 1>(strides2.begin() + 1, N - 1));
    dst += strides1[0];
    src += strides2[0];
  }
}


template <Arithmetic T>
inline void
initialize_unsafe(T* dst, std::span<const size_t, 0>, std::span<const ssize_t, 0>, T init)
{
  *dst = init;
}

template <Arithmetic T>
inline void
initialize_unsafe(T* dst, std::span<const size_t, 1> dimensions, std::span<const ssize_t, 1> strides, T init)
{
  for (size_t i = 0; i < dimensions[0]; i++, reinterpret_cast<char*&>(dst) += strides[0])
    *dst = init;
}

template <Arithmetic T, size_t N>
inline void
initialize_unsafe(T* dst, std::span<const size_t, N> dimensions, std::span<const ssize_t, N> strides, T init)
{
  for (size_t i = 0; i < dimensions[0]; i++, reinterpret_cast<char*&>(dst) += strides[0])
    initialize_unsafe(dst,
                      std::span<const size_t, N - 1>(dimensions.begin() + 1, dimensions.end()),
                      std::span<const ssize_t, N - 1>(strides.begin() + 1, strides.end()),
                      init);
}

template <Arithmetic T>
inline void initialize_unsafe(T* dst, size_t size, T init)
{
  for (size_t i = 0; i < size; i++)
    *dst++ = init;
}

} // end of namespace details

/// Array manages a buffer of elements of a specific type.
///
/// The buffer can be statically or dynamically allocated, and in system memory or device memory.
/// The array size defines the number of elements and is not the size in bytes.
template <typename T, typename TAllocator>
class Array
{
  template <typename, typename> friend class Array;

  using value_type = T;
  using allocator_type = TAllocator;
  using pointer = std::allocator_traits<TAllocator>::pointer;
  using const_pointer = std::allocator_traits<TAllocator>::const_pointer;

 public:
  Array() = default;

  ~Array()
  {
    if (pointer_ != nullptr)
      allocator_.deallocate(pointer_, size_);
  }

  // @brief Allocates a buffer of the provided size.
  explicit Array(size_t size) : size_(size)
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
  Array(const Array& other) : size_(other.Size())
  {
    pointer_ = allocator_.allocate(size_);
    std::memcpy(std::to_address(pointer_), std::to_address(other.pointer_), other.Size() * sizeof(value_type));
  }

  // @brief Move constructor.
  Array(Array&& other)
    : size_(other.Size()),
      pointer_(std::move(other.pointer_))
  {
    other.pointer_ = nullptr;
  }

  // @brief Constructor from arrays with different allocators
  template <typename A>
  Array(const Array<value_type, A>& other) : size_(other.Size())
  {
    pointer_ = allocator_.allocate(size_);
    std::memcpy(std::to_address(pointer_), other.Data(), size_ * sizeof(value_type));
  }

  /// @brief Move assignement
  Array& operator=(Array&& other)
  {
    if (pointer_ != nullptr)
      allocator_.deallocate(pointer_, size_);

    size_ = other.Size();
    pointer_ = std::move(other.pointer_);
    other.pointer_ = nullptr;

    return *this;
  }

  template <typename A>
  Array& operator=(const Array<value_type, A>& other)
  {
    if (other.Size() != size_)
    {
      allocator_.deallocate(pointer_, size_);
      allocator_.allocate(other.Size());
      size_ = other.Size();
    }
    std::memcpy(std::to_address(pointer_), other.Data(), size_ * sizeof(value_type));

    return *this;
  }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return size_; }

  /// Data returns a pointer to the data buffer.
  value_type* Data()                                      { return &*pointer_; }

  /// Data returns a pointer to the data buffer.
  const value_type* Data() const                          { return &*pointer_; }

  // TODO: temporary addition to support Metal buffers

  // Buffer returns the MTL buffer - internal use only
  auto Buffer()                                           { return pointer_.Buffer(); }

  // Buffer returns the MTL buffer - internal use only
  auto Buffer() const                                     { return pointer_.Buffer(); }

 private:
  allocator_type  allocator_;
  size_t          size_;
  pointer         pointer_;
};


/// Array specialization for storing a single scalar
template <Arithmetic T>
class Array<T, Scalar>
{
 public:
  using value_type = T;
  using pointer = value_type*;
  using const_pointer = const value_type*;

 public:
  Array() = default;

  // @brief Initializes the data value to init.
  Array(size_t s, value_type init) : data_(init)
  {
    if (s != 1)
      throw std::runtime_error("internal error: invalid size for Array<Scalar>");
  }

  Array(size_t s)
  {
    if (s != 1)
      throw std::runtime_error("internal error: invalid size for Array<Scalar>");
  }

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return 1UL; }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return &data_; }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return &data_; }

 private:
  value_type  data_;
};


/// Array specialization for static data.
template <Arithmetic T, size_t... Ns>
class Array<T, StaticResource<Ns...>>
{
 public:
  using value_type = T;
  using pointer = const value_type*;
  using const_pointer = const value_type*;
  static constexpr size_t size = (... * Ns);


 public:
  // @brief Iniitializes a constant array
  Array(std::array<T, size>&& array) : array_(array) {}

  // Explicity disallow default and copy constructors for StaticResource arrays.
  Array() = delete;
  Array(const Array& other) = delete;

  // Support move construction (TODO: revisit)
  Array(Array&& other) : array_(std::move(other.array_)) {}

  // Explicitly disallow copy and move assign operators for StaticResource arays.
  Array& operator=(Array&& other) = delete;
  Array& operator=(const Array& other) = delete;

  /// Size returns the size of the entire buffer.
  constexpr size_t Size() const                           { return size; }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return array_.data(); }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return array_.data(); }

 private:
  const std::array<value_type, size>  array_;
};


// Arrray specialization for MemoryMapped memory
template <typename T>
class Array<T, MemoryMapped>
{
  using value_type = T;
  using pointer = const value_type*;
  using const_pointer = const value_type*;

 public:
  // @brief Constructor for a memory mapped area
  Array(value_type* data, size_t size) : pointer_(data), size_(size) {}

  // Explicity disallow default, copy, and move constructors for static memory arrays.
  Array() = delete;
  Array(const Array& other) = delete;
  Array(Array&& other) = delete;

  // Explicitly disallow copy and move assign operators for static memory arays.
  Array& operator=(Array&& other) = delete;
  Array& operator=(const Array& other) = delete;

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return size_; }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return pointer_; }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return pointer_; }

 private:
  pointer       pointer_;
  const size_t  size_;
};


// Array specialization for a view
template <typename T, typename Allocator>
class Array<T, View<Allocator>>
{
  using array_type = Array<T, Allocator>;
  using value_type = T;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using allocator_type = Allocator;

 public:

  Array() = delete;
  ~Array() = default;

  // @brief Constructor for a view from an array.
  Array(Array<value_type, allocator_type>& array, size_t size, size_t offset)
    : array_(array), size_(size), offset_(offset)
  {}

  Array(Array<value_type, allocator_type>&& array, size_t size, size_t offset)
    : array_(std::move(array)), size_(size), offset_(offset)
  {}

  // @brief Copy constructor.
  Array(const Array& other)
    : array_(other.array_), size_(other.Size()), offset_(other.offset_)
  {}

  // @brief Move constructor.
  Array(Array&& other)
    : array_(other.array_), size_(other.Size()), offset_(other.offset_)
  {}


  Array& operator=(const Array<value_type, allocator_type>& other)
  {
    array_ = other.array_;
    size_ = other.Size();
    offset_ = other.offset_;
  }

  /// Size returns the size of the view.
  size_t Size() const                                     { return size_; }

  /// Data returns a pointer to the data buffer.
  value_type* Data()                                      { return array_.Data(); }

  /// Data returns a pointer to the data buffer.
  const value_type* Data() const                          { return array_.Data(); }

  // Buffer returns the MTL buffer - internal use only
  auto Buffer()                                           { return array_.Buffer(); }

  // Buffer returns the MTL buffer - internal use only
  auto Buffer() const                                     { return array_.Buffer(); }


 private:
  array_type& array_;
  size_t      size_;
  size_t      offset_;
};

// CTAD rules (gcc 13.3.0 get's confused)
template <typename T, typename Alloc> Array(const Array<T, Alloc>&) -> Array<T, Alloc>;
template <typename T, typename Alloc> Array(Array<T, Alloc>&&) -> Array<T, Alloc>;


} // end of namespace libai

#endif // LIBAI_TENSOR_ARRAY_H
