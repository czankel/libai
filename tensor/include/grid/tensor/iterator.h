//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef LIBAI_TENSOR_TENSOR_ITERATOR_H
#define LIBAI_TENSOR_TENSOR_ITERATOR_H

namespace libai {
namespace details {

class Tensor;

// TODO: ConstIterator is mostly a duplicate for Iterator but for const_pointer
// TODO: operator++/-- are slow; can optimize by folding indices

/// Iterator implements a bidirection iterator for tensors.
template <typename TTensor>
class Iterator
{
 public:
  using iterator_category = std::bidirectional_iterator_tag;
  using difference_type   = std::ptrdiff_t;
  using value_type        = typename TTensor::value_type;
  using pointer           = typename TTensor::pointer;
  using reference         = typename TTensor::reference;
  static constexpr size_t rank = TTensor::rank;

  Iterator() = default;

  // Note that ranges iterators must be default-initializable, so cannot use reference for Tensor
  Iterator(TTensor& tensor) : coordinates_{}, tensor_(&tensor), index_(0) {}

  Iterator(TTensor& tensor, const std::array<size_t, rank>& dimensions)
    : coordinates_(dimensions),
      tensor_(&tensor),
      index_(0)
  {
    auto& strides = tensor_->Strides();
    for (ssize_t axis = rank-1; axis >= 0; axis--)
      index_ += coordinates_[axis] * strides[axis];
  }

  // TODO provide Iterator for initializatin with 'coordinates'?

  reference operator*() const                     { return tensor_->Data()[index_]; }

  Iterator& operator++()
  {
    auto& extents = tensor_->Dimensions();
    auto& strides = tensor_->Strides();

    // most-common case
    ssize_t axis = rank - 1;
    index_ += strides[axis];
    if (++coordinates_[axis] < extents[axis])
      return *this;

    coordinates_[axis--] = 0;
    for (; axis >= 0 && ++coordinates_[axis] == extents[axis]; axis--)
      coordinates_[axis] = 0;

    if (axis == -1)
      throw std::runtime_error("index out of bounds");

    for (index_ = 0; axis >= 0; axis--)
      index_ += coordinates_[axis] * strides[axis];

    return *this;
  }

  Iterator& operator--()
  {
    auto& extents = tensor_->Dimensions();
    auto& strides = tensor_->Strides();

    // most-common case
    ssize_t axis = rank - 1;
    index_ += strides[axis];
    if (--coordinates_[axis]-- != 0)
      return *this;

    coordinates_[axis] = extents[axis] - 1;
    axis--;
    for (; axis >= 0 && coordinates_[axis]-- == 0; axis--)
      coordinates_[axis] = extents[axis] - 1;

    if (axis == -1)
      throw std::runtime_error("index out of bounds");

    for (index_ = 0; axis >= 0; axis--)
      index_ += coordinates_[axis] * strides[axis];

    return *this;
  }


  Iterator operator++(int)                        { Iterator tmp = *this; ++(*this); return tmp; }
  Iterator operator--(int)                        { Iterator tmp = *this; --(*this); return tmp; }


  /// @brief return a new iterator with added distances to the positions for the provided dimensions.
  template <size_t S>
  Iterator operator+(const std::array<size_t, S>& dist)
  {
    Iterator tmp = *this;
    for (size_t i = 0; i < S; i++)
      tmp.coordinates_[i] += dist[i];
    return tmp;
  }

  /// @brief add distances to the positions of the iterator for the provided dimensions.
  template <size_t S>
  Iterator& operator+=(const std::array<size_t, S>& dist)
  {
    for (size_t i = 0; i < S; i++)
      coordinates_[i] += dist[i];
    return *this;
  }

  friend bool operator==(const Iterator& a, const Iterator& b) { return a.index_ == b.index_; }

  // iterator extensions
  constexpr size_t                  Rank() const        { return rank; }
  const std::array<size_t, rank>&   Extents() const     { return tensor_->Dimensions(); }
  const std::array<ssize_t, rank>&  Strides() const     { return tensor_->Strides(); }
  const std::array<size_t, rank>&   Coordinates() const { return coordinates_; }
  auto                              Buffer()            { return tensor_->Buffer(); }
  size_t                            Offset() const      { return tensor_->Offset(); }

 private:
  std::array<size_t, rank>  coordinates_;
  TTensor*                  tensor_;
  size_t                    index_;
};


/// ConstIterator implements a bidirection iterator for tensors.
template <typename TTensor>
class ConstIterator
{
 public:
  using iterator_category = std::bidirectional_iterator_tag;
  using difference_type   = std::ptrdiff_t;
  using value_type        = typename TTensor::value_type;
  using const_pointer     = typename TTensor::const_pointer;
  using const_reference   = typename TTensor::const_reference;
  static constexpr size_t rank = TTensor::rank;

  ConstIterator() = default;

  ConstIterator(const TTensor& tensor) : coordinates_{}, tensor_(&tensor), index_(0) {}

  ConstIterator(const TTensor& tensor, const std::array<size_t, rank>& dimensions)
    : coordinates_(dimensions),
      tensor_(&tensor),
      index_(0)
  {
    auto& strides = tensor_->Strides();
    for (ssize_t axis = rank-1; axis >= 0; axis--)
      index_ += coordinates_[axis] * strides[axis];
  }


  const_reference operator*() const               { return tensor_->Data()[index_]; }

  ConstIterator& operator++()
  {
    auto& extents = tensor_->Dimensions();
    auto& strides = tensor_->Strides();

    // most-common case
    ssize_t axis = rank - 1;
    index_ += strides[axis];
    if (++coordinates_[axis] < extents[axis])
      return *this;

    coordinates_[axis--] = 0;
    for (; axis >= 0 && ++coordinates_[axis] == extents[axis]; axis--)
      coordinates_[axis] = 0;

    if (axis == -1)
      throw std::runtime_error("index out of bounds");

    for (index_ = 0; axis >= 0; axis--)
      index_ += coordinates_[axis] * strides[axis];

    return *this;
  }

  ConstIterator& operator--()
  {
    auto& extents = tensor_->Dimensions();
    auto& strides = tensor_->Strides();

    // most-common case
    ssize_t axis = rank - 1;
    index_ += strides[axis];
    if (--coordinates_[axis]-- != 0)
      return *this;

    coordinates_[axis] = extents[axis] - 1;
    axis--;
    for (; axis >= 0 && coordinates_[axis]-- == 0; axis--)
      coordinates_[axis] = extents[axis] - 1;

    if (axis == -1)
      throw std::runtime_error("index out of bounds");

    for (index_ = 0; axis >= 0; axis--)
      index_ += coordinates_[axis] * strides[axis];

    return *this;
  }


  ConstIterator operator++(int)                  { ConstIterator tmp = *this; ++(*this); return tmp; }
  ConstIterator operator--(int)                  { ConstIterator tmp = *this; --(*this); return tmp; }


  /// @brief return a new iterator with added distances to the positions for the provided dimensions.
  template <size_t S>
  ConstIterator operator+(const std::array<size_t, S>& dist)
  {
    ConstIterator tmp = *this;
    for (size_t i = 0; i < S; i++)
      tmp.coordinates_[i] += dist[i];
    return tmp;
  }

  /// @brief add distances to the positions of the iterator for the provided dimensions.
  template <size_t S>
  ConstIterator& operator+=(const std::array<size_t, S>& dist)
  {
    for (size_t i = 0; i < S; i++)
      coordinates_[i] += dist[i];
    return *this;
  }

  friend bool operator==(const ConstIterator& a, const ConstIterator& b) { return a.index_ == b.index_; }

  // iterator extensions
  constexpr size_t                  Rank() const        { return rank; }
  const std::array<size_t, rank>&   Extents() const     { return tensor_->Dimensions(); }
  const std::array<ssize_t, rank>&  Strides() const     { return tensor_->Strides(); }
  const std::array<size_t, rank>&   Coordinates() const { return coordinates_; }
  auto                              Buffer()            { return tensor_->Buffer(); }
  size_t                            Offset() const      { return tensor_->Offset(); }

 private:
  std::array<size_t, rank>  coordinates_;
  const TTensor*            tensor_;
  size_t                    index_;
};


} // end of namespace details
} // end of namespace libai

#endif  // LIBAI_TENSOR_TENSOR_ITERATOR_H
