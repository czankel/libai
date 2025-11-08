//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef LIBAI_UTIL_DEMANGLE_H
#define LIBAI_UTIL_DEMANGLE_H

#include <cxxabi.h>
#include <stdlib.h>

#include <string>

namespace libai {


inline std::string Demangle(const std::string& mangled_name)
{
  int status = -4;
  char* res = abi::__cxa_demangle(mangled_name.c_str(), NULL, NULL, &status);
  const char* const demangled_name = (status == 0) ? res : mangled_name.c_str();
  std::string name(demangled_name);
  free(res);
  return name;
}


} // end of namespace libai

#endif  // LIBAI_UTIL_DEMANGLE_H
