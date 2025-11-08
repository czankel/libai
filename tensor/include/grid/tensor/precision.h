//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#ifndef GRID_TENSOR_PRECISION_H
#define GRID_TENSOR_PRECISION_H

namespace libai {

template <typename> struct Eps {};
template <> struct Eps<float>  { constexpr static float  default_value = 1e-5f; float  value; };
template <> struct Eps<double> { constexpr static double default_value = 1e-5f; double value; };


/// Usage:
///   Precision(margin) or Set(margin-margin)
///   Epsilon() returns ...
///   Precision::Reset();
///   Precision::xx(1e-6);
///   Precision::Eps<value_type>() returns the ...
///
/// {
///   Precision(10f);
///   bool equals = (tensor1 == tensor2);
/// }

class Precision
{
 public:
  Precision(float margin)           { g_margin_ = margin; }
  Precision()                       { Reset(); }
  ~Precision()                      { Reset(); }

  void Set(float margin)            { g_margin_ = margin; }
  void Reset()                      { g_margin_ = 1.f; }

  static float Margin()             { return g_margin_; }

 private:
  static __thread float g_margin_;
};

} // end of namespace libai


#endif  // GRID_TENSOR_PRECISION_H
