# Copyright (C) Chris Zankel. All rights reserved.
# This code is subject to U.S. and other copyright laws and
# intellectual property protections.
#
# The contents of this file are confidential and proprietary to Chris Zankel.

add_subdirectory(source)

if(BUILD_TEST)
  add_subdirectory(unittest)
endif()
