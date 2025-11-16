//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef LIBAI_TENSOR_ARRAY_H
#define LIBAI_TENSOR_ARRAY_H

#include <span>

#include "concepts.h"
#include "memory.h"

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
template <typename, typename> class Array;


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

 protected:
  value_type  data_;
};


/// Array specialization for static data.
template <Arithmetic T, size_t... Ns>
class Array<T, StaticMemory<Ns...>>
{
 public:
  using value_type = T;
  using pointer = const value_type*;
  using const_pointer = const value_type*;
  static constexpr size_t size = (... * Ns);


 public:
  // @brief Iniitializes a constant array
  Array(std::array<T, size>&& array) : array_(array) {}

  // Explicity disallow default, copy, and move constructors for StaticMemory arrays.
  Array() = delete;
  Array(const Array& other) = delete;
  Array(Array&& other) = delete;

  // Explicitly disallow copy and move assign operators for StaticMemory arays.
  Array& operator=(Array&& other) = delete;
  Array& operator=(const Array& other) = delete;

  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return size; }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return array_.data(); }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return array_.data(); }

 protected:
  const std::array<value_type, size>  array_;
};


} // end of namespace libai

#endif // LIBAI_TENSOR_ARRAY_H
