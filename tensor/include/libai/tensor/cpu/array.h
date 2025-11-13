//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef LIBAI_TENSOR_CPU_ARRAY_H
#define LIBAI_TENSOR_CPU_ARRAY_H

#include <stdexcept>

#include "device.h"

#include "../array.h"
#include "../tensor_parameters.h"

namespace libai {

/// brief: Array is a specialization for a dynamically allocated buffer.
template <typename T>
class Array<T, DeviceMemory<device::CPU>>
{
  using value_type = T;
  using pointer = value_type*;
  using const_pointer = const value_type*;

 public:
  Array() = default;

  // @brief Constructor for a contiguous array with the provided size (number of elements)
  Array(size_t size)
    : size_(size),
      data_(static_cast<pointer>(operator new[](size_ * sizeof(T), std::align_val_t(16))))
  {}

  // @brief Constructor for a contiguous array with the provided size (number of elements) with initialization.
  Array(size_t size, value_type init)
    : size_(size),
      data_(static_cast<pointer>(operator new[](size_ * sizeof(T), std::align_val_t(16))))
  {
    details::initialize_unsafe(data_, size_, init);
  }

  // @brief Constructor for a non-contiguous array with the provided dimensions and strides.
  template <size_t N>
  Array(const std::array<size_t, N>& dimensions, const std::array<ssize_t, N>& strides)
    : size_(get_array_size<value_type>(dimensions, strides)),
      data_(static_cast<pointer>(operator new[](size_ * sizeof(T), std::align_val_t(16))))
  {}

  // @brief Constructor for a non-contiguous array with the provided dimensions and strides with initialization.
  template <size_t N>
  Array(const std::array<size_t, N>& dimensions, const std::array<ssize_t, N>& strides, value_type init)
    : size_(get_array_size<value_type>(dimensions, strides)),
      data_(static_cast<pointer>(operator new[](size_ * sizeof(T), std::align_val_t(16))))
  {
    details::initialize_unsafe(data_, dimensions, strides, init);
  }

  // @brief Move constructor.
  Array(Array&& other) : size_(other.size_), data_(std::move(other.data_)) { other.data_ = nullptr; }

  // @brief Copy constructor of contiguous arrays.
  Array(const Array& other)
    : size_(other.size_),
      data_(static_cast<pointer>(operator new[](size_ * sizeof(T), std::align_val_t(16))))
  {
    memcpy(data_, other.data_, other.size_ * sizeof(T));
  }

  // FIXME: conversion, i.e. int -> float

  // @brief Copy constructor from same array type with dimensions and strides
  template <size_t N>
  Array(const Array& other,
        const std::array<size_t, N>& dimensions,
        const std::array<ssize_t, N>& strides1,
        const std::array<ssize_t, N>& strides2)
    : size_(get_array_size<value_type>(dimensions, strides1)),
      data_(static_cast<pointer>(operator new[](size_ * sizeof(T), std::align_val_t(16))))
  {
    details::copy_unsafe(data_, other.Data(),
                         std::span<const size_t, N>(dimensions.begin(), N),
                         std::span<const ssize_t, N>(strides1.begin(), N),
                         std::span<const ssize_t, N>(strides2.begin(), N));
  }

  template <typename S, size_t N>
  Array(const Array<S, DeviceMemory<device::CPU>>& other,
        const std::array<size_t, N>& dimensions,
        const std::array<ssize_t, N>& strides1,
        const std::array<ssize_t, N>& strides2)
    : size_(get_array_size<value_type>(dimensions, strides1)),
      data_(static_cast<pointer>(operator new[](size_ * sizeof(T), std::align_val_t(16))))
  {
    details::copy_unsafe(data_, other.Data(),
                         std::span<const size_t, N>(dimensions.begin(), N),
                         std::span<const ssize_t, N>(strides1.begin(), N),
                         std::span<const ssize_t, N>(strides2.begin(), N));
  }

  // @brief Copy constructor from different array type with dimensions and strides
  template <typename S, size_t N>
  Array(const S* data,
        const std::array<size_t, N>& dimensions,
        const std::array<ssize_t, N>& strides1,
        const std::array<ssize_t, N>& strides2)
    : size_(get_array_size<value_type>(dimensions, strides1)),
      data_(static_cast<pointer>(operator new[](size_ * sizeof(T), std::align_val_t(16))))
  {
    printf("ARRAY from POINTER\n");
    details::copy_unsafe(data_, data,
                         std::span<const size_t, N>(dimensions.begin(), N),
                         std::span<const ssize_t, N>(strides1.begin(), N),
                         std::span<const ssize_t, N>(strides2.begin(), N));
  }


  ~Array()
  {
    if (data_ != nullptr)
      operator delete[](data_, std::align_val_t(16));
  }

  Array& operator=(Array&& other)
  {
    if (data_ != nullptr)
      operator delete[](data_, std::align_val_t(16));

    size_ = other.size_;
    data_ = std::move(other.data_);
    other.data_ = nullptr;

    return *this;
  }

  // Disable copy assignment FIXME: may need to support, and have option with strides
  Array& operator=(const Array& other) = delete;


  /// Resize resizes the buffer of the Array. This will destroy the current buffer and return
  /// an uninitialized buffer of the new size.
  Array& Realloc(size_t size)
  {
    if (size != size_)
    {
      if (data_ != nullptr)
        operator delete[](data_, std::align_val_t(16));
      data_ = static_cast<pointer>(operator new[](size * sizeof(T), std::align_val_t(16)));
      size_ = size;
    }

    return *this;
  }


  /// Size returns the size of the entire buffer.
  size_t Size() const                                     { return size_; }

  /// Data returns a pointer to the data buffer.
  pointer Data()                                          { return data_; }

  /// Data returns a pointer to the data buffer.
  const_pointer Data() const                              { return data_; }

 protected:
  size_t  size_;
  pointer data_;
};


} // end of namespace libai

#endif  // LIBAI_TENSOR_CPU_ARRAY_H
