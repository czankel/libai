# Copyright (C) Chris Zankel. All rights reserved.
# This code is subject to U.S. and other copyright laws and
# intellectual property protections.
#
# The contents of this file are confidential and proprietary to Chris Zankel.

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# Bad variable reference syntax is an error.
# https://cmake.org/cmake/help/latest/policy/CMP0010.html
cmake_policy(SET CMP0010 NEW)

# Compiler id for Apple Clang is now AppleClang.
# https://cmake.org/cmake/help/latest/policy/CMP0025.html
cmake_policy(SET CMP0025 NEW)

##
## Project
##

project(gridtensor CXX)

##
## Project Options and configurations
## (Ensure to also update cmake/Summary.cmake)
##

if (APPLE)
  option(BUILD_METAL "Build with support for Apple's Metal" ON)
endif()

option(BUILD_TEST "Build C++ test binaries (need gtest and gbenchmark)" OFF)
option(BUILD_DOCS "Build documentation" OFF)
if (LINUX)
  option(BUILD_DEB_PACKAGE "Build a debian package" OFF)
  option(BUILD_RPM_PACKAGE "Build an RPM package" OFF)
endif()
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_STATIC_LIBS "Build static libraries" OFF)

option(BUILD_CUDA "Build CUDA support" OFF)

option(USE_COLORIZE_OUTPUT "Colorize output during compilation" ON)
option(USE_CPP_CODE_COVERAGE "Compile C/C++ with code coverage flags" OFF)
option(USE_CCACHE "" OFF)
option(USE_GOLD_LINKER "Use gold linkder" OFF)

##
## Include helper functions
##

include(cmake/utils.cmake)


##
## Compiler configuration and flags
##

if(env_cxx_standard GREATER -1)
  message(
      WARNING "C++ standard version definition detected in environment variables."
      "GridStreamer requires -std=c++20. Please remove -std=c++ settings from your environment.")
endif()


# CXX/C base configuration

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard target property")
set(CMAKE_C_STANDARD   11 CACHE STRING "C standard target property")

# CMAKE configuration

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INSTALL_MESSAGE NEVER)

# CCache support

if(USE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C compiler launcher")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CXX compiler launcher")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CUDA compiler launcher")
  else()
    message(STATUS "Could not find ccache. Consider installing ccache to speed up compilation.")
  endif()
endif()

# Pthreads

if(NOT ANDROID)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
endif()

# Compiler flags

if(NOT MSVC)
  string(APPEND CMAKE_CXX_FLAGS " -O3 -fPIC")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-narrowing")
  # Eigen fails to build with some versions, so convert this to a warning
  # Details at http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1459
  string(APPEND CMAKE_CXX_FLAGS " -Wall")
  string(APPEND CMAKE_CXX_FLAGS " -Wextra")
  check_cxx_compiler_flag_and_append("-Werror=return-type")
  check_cxx_compiler_flag_and_append("-Werror=non-virtual-dtor")
  check_cxx_compiler_flag_and_append("-Werror=braced-scalar-init")
  check_cxx_compiler_flag_and_append("-Werror=range-loop-construct")
  check_cxx_compiler_flag_and_append("-Wno-missing-field-initializers")
  check_cxx_compiler_flag_and_append("-Wno-type-limits")
  check_cxx_compiler_flag_and_append("-Wno-array-bounds")
  check_cxx_compiler_flag_and_append("-Wno-unknown-pragmas")
  check_cxx_compiler_flag_and_append("-Wunused-local-typedefs")
  check_cxx_compiler_flag_and_append("-Wno-unused-parameter")
  check_cxx_compiler_flag_and_append("-Wno-unused-function")
  check_cxx_compiler_flag_and_append("-Wno-unused-result")
  check_cxx_compiler_flag_and_append("-Wno-strict-overflow")
  check_cxx_compiler_flag_and_append("-Wno-strict-aliasing")
  check_cxx_compiler_flag_and_append("-Wno-error=deprecated-declarations")
  check_cxx_compiler_flag_and_append("-Wvla-extension")
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-range-loop-analysis")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-pass-failed")
  endif()
  if(CMAKE_COMPILER_IS_GNUCXX AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0.0))
    # Suppress issue: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=43407
    string(APPEND CMAKE_CXX_FLAGS " -Wno-attributes")
  endif()
  if(CMAKE_COMPILER_IS_GNUCXX AND NOT (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0.0))
    string(APPEND CMAKE_CXX_FLAGS " -Wno-stringop-overflow")
  endif()
  if(CMAKE_COMPILER_IS_GNUCXX)
    # Suppress "The ABI for passing parameters with 64-byte alignment has changed in GCC 4.6"
    string(APPEND CMAKE_CXX_FLAGS " -Wno-psabi")
  endif()

  # Use ld.gold if available, fall back to ld.bfd (the default ld) if not
  if(USE_GOLD_LINKER)
    execute_process(
      COMMAND
      "${CMAKE_C_COMPILER}" -fuse-ld=gold -Wl,--version
       ERROR_QUIET
       OUTPUT_VARIABLE LD_VERSION)
    if(NOT "${LD_VERSION}" MATCHES "GNU gold")
      message(WARNING "USE_GOLD_LINKER was set but ld.gold isn't available, turning it off")
      set(USE_GOLD_LINKER OFF)
    else()
      message(STATUS "ld.gold is available, using it to link")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=gold")
      set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fuse-ld=gold")
    endif()
  endif()

  check_cxx_compiler_flag_and_append("-Wno-error=pedantic")
  check_cxx_compiler_flag_and_append("-Wno-error=redundant-decls")
  check_cxx_compiler_flag_and_append("-Wno-error=old-style-cast")
  # These flags are not available in GCC-4.8.5. Set only when using clang.
  # Compared against https://gcc.gnu.org/onlinedocs/gcc-4.8.5/gcc/Option-Summary.html
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    check_cxx_compiler_flag_and_append("-Wconstant-conversion")
    check_cxx_compiler_flag_and_append("-Wno-invalid-partial-specialization")
    check_cxx_compiler_flag_and_append("-Wno-typedef-redefinition")
    check_cxx_compiler_flag_and_append("-Wno-unused-private-field")
    check_cxx_compiler_flag_and_append("-Wno-inconsistent-missing-override")
    check_cxx_compiler_flag_and_append("-Wno-aligned-allocation-unavailable")
    check_cxx_compiler_flag_and_append("-Wno-c++14-extensions")
    check_cxx_compiler_flag_and_append("-Wno-c++17-extensions")
    check_cxx_compiler_flag_and_append("-Wno-constexpr-not-const")
    check_cxx_compiler_flag_and_append("-Wno-missing-braces")
    check_cxx_compiler_flag_and_append("-Wunused-lambda-capture")
    check_cxx_compiler_flag_and_append("-Wunused-local-typedef")
    check_cxx_compiler_flag_and_append("-Qunused-arguments")
    if(${USE_COLORIZE_OUTPUT})
    endif()
  endif()

  if(${USE_COLORIZE_OUTPUT})
    check_cxx_compiler_flag_and_append("-fcolor-diagnostics")
    check_cxx_compiler_flag_and_append("-fdiagnostics-color=always")
  endif()

  if((APPLE AND (NOT ("${CLANG_VERSION_STRING}" VERSION_LESS "9.0")))
    OR(CMAKE_COMPILER_IS_GNUCXX
    AND(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0 AND NOT APPLE)))
    string(APPEND CMAKE_CXX_FLAGS " -faligned-new")
  endif()
  if(WERROR)
    check_cxx_compiler_flag("-Werror" COMPILER_SUPPORT_WERROR)
    if(NOT COMPILER_SUPPORT_WERROR)
      set(WERROR FALSE)
    else()
      string(APPEND CMAKE_CXX_FLAGS " -Werror")
    endif()
  endif(WERROR)
  if(NOT APPLE)
    check_cxx_compiler_flag_and_append("-Wno-unused-but-set-variable")
    check_cxx_compiler_flag_and_append("-Wno-maybe-uninitialized")
  endif()
  string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fno-omit-frame-pointer -g -O3")

  string(APPEND CMAKE_LINKER_FLAGS_DEBUG " -fno-omit-frame-pointer -O0")
  # -ldflags="-X github.com/czankel//config.CneVersion=${VERSION}" FIXME.. or --defsym? -X is not default??
  check_cxx_compiler_flag_and_append("-fno-math-errno")
  check_cxx_compiler_flag_and_append("-fno-trapping-math")
  check_cxx_compiler_flag_and_append("-Werror=format")
  ## Foundation/NSObject.hpp:214:39: error: cast from 'void (*)()' to 'SendMessageProc'
  if(NOT APPLE)
    check_cxx_compiler_flag_and_append("-Werror=cast-function-type")
  endif()
endif()

##
## Include Test (GTest) if configured
##

if (BUILD_TEST)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
  )

  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(BUILD_GTEST OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

  FetchContent_MakeAvailable(googletest)
endif()

##
## Include Metal libraries if configured
##

if (BUILD_METAL)

  execute_process(COMMAND zsh "-c" "/usr/bin/xcrun -sdk macosx --show-sdk-version"
                  OUTPUT_VARIABLE MACOS_VERSION
                  COMMAND_ERROR_IS_FATAL ANY)

  message(STATUS "Building with SDK for macOS version ${MACOS_VERSION}")

  if (${MACOS_VERSION} GREATER_EQUAL 14.2)
    set(METAL_CPP_URL https://developer.apple.com/metal/cpp/files/metal-cpp_macOS14.2_iOS17.2.zip)
  elseif (${MACOS_VERSION} GREATER_EQUAL 14.0)
    set(METAL_CPP_URL https://developer.apple.com/metal/cpp/files/metal-cpp_macOS14_iOS17-beta.zip)
  elseif (${MACOS_VERSION} GREATER_EQUAL 13.3)
    set(METAL_CPP_URL https://developer.apple.com/metal/cpp/files/metal-cpp_macOS13.3_iOS16.4.zip)
  else()
    message(FATAL_ERROR "MLX requires macOS >= 13.4 to be built with MLX_BUILD_METAL=ON" )
  endif()

  option(CMAKE_TLS_VERIFY "Verify SSL certificates" ON)
  set_property(DIRECTORY PROPERTY EP_UPDATE_DISCONNECTED true)

  message(STATUS "Building METAL sources")
  add_compile_definitions(_METAL_)

  FetchContent_Declare(
    metal_cpp
    URL ${METAL_CPP_URL}
    DOWNLOAD_EXTRACT_TIMESTAMP NEW
  )

  FetchContent_MakeAvailable(metal_cpp)

  find_library(METAL_LIB Metal)
  find_library(FOUNDATION_LIB Foundation)
  find_library(QUARTZ_LIB QuartzCore)

endif()


##
## Add CMakeLists from sub-directories
##

add_subdirectory(tensor)
add_subdirectory(util)

set(gridtensor_HEADER_DIRS
  "tensor/include"
  "models/include"	# FIXME
  "util/include"	# FIXME
)

##
## Target libraries
##

get_property(gridtensor_SOURCES GLOBAL PROPERTY gridtensor_SOURCES)

if (BUILD_SHARED_LIBS AND NOT ANDROID)
  add_library(gridtensor SHARED ${gridtensor_SOURCES})
endif()

if (BUILD_STATIC_LIBS)
  add_library(gridtensor_static STATIC ${gridtensor_SOURCES})
endif()

if (UNIX AND NOT APPLE)
  set (EXTRA_LIBS ${EXTRA_LIBS} rt)
endif()

target_include_directories(gridtensor PUBLIC ${gridtensor_HEADER_DIRS})

##
## Enable CUDA
##

if (BUILD_CUDA)
  set_target_properties(gridtensor PROPERTIES CUDA_ARCHITECTURES "75")
  set_target_properties(gridtensor PROPERTIES CUDA_STANDARD 20)
  enable_language(CUDA)
  target_compile_definitions(gridtensor PUBLIC BUILD_CUDA=1)
endif()

##
## Build Metal sources
##

if (BUILD_METAL)

  target_include_directories(
    gridtensor PUBLIC
    $<BUILD_INTERFACE:${metal_cpp_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/metal_cpp>
  )

  target_link_libraries(gridtensor ${METAL_LIB} ${FOUNDATION_LIB} ${QUARTZ_LIB})

  get_property(gridtensor_metal_PATH GLOBAL PROPERTY gridtensor_metal_PATH)
  target_compile_definitions(
    gridtensor PUBLIC METAL_PATH="${gridtensor_metal_PATH}/gridtensor.metallib" BUILD_METAL=1
  )

  add_dependencies(
    gridtensor
    gridtensor_metallib
  )

endif()

##
## Target Tools
##

add_executable(llama tools/llama.cc models/llama/llama.cc models/llama/karpathy.cc models/llama/ggml.cc)
target_include_directories(llama PUBLIC ${gridtensor_HEADER_DIRS})
target_link_libraries(llama gridtensor)

##
## Install libraries and header files
##

if (BUILD_SHARED_LIBS)
  install(TARGETS gridtensor LIBRARY DESTINATION lib)
endif()

if (BUILD_STATIC_LIBS)
  install(TARGETS gridtensor_static ARCHIVE DESTINATION lib)
endif()

foreach(dir ${gridtensor_HEADER_DIRS})
  install(DIRECTORY ${dir}/ DESTINATION include)
endforeach()

# FIXME: use forach(... ${gridtensor_TOOLS})
install(TARGETS llama DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

#FIXME
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin/")
#include_directories("${CMAKE_INSTALL_PREFIX}/include")
#link_directories("${CMAKE_INSTALL_PREFIX}/lib")

##
## Android Project
##

# Copy header files and libraries to the optionally provided project directory
#  <project>/grid_sdk/android/include/
#  <project>/grid_sdk/android/lib/<platform>/
#
# The following two variables need to be defined:
#  ANDROID_SDK_DIR pointing to the gridtensor sdk destination directory
#  ANDROID_PLATFORM platform, such as armeabi-v7a, etc.


if (ANDROID_SDK_DIR AND ANDROID_PLATFORM)
  install(TARGETS gridtensor_static ARCHIVE DESTINATION ${ANDROID_SDK_DIR}/lib/${ANDROID_PLATFORM})
  install(DIRECTORY grid/include/grid DESTINATION ${ANDROID_SDK_DIR}/include)
endif()

##
## GoogleTest
##

if (BUILD_TEST)
  get_property(gridtensor_test_SOURCES GLOBAL PROPERTY gridtensor_test_SOURCES)
  add_executable(gridtensor_test ${gridtensor_test_SOURCES})
  target_link_libraries(gridtensor_test gridtensor gtest gtest_main)
  target_include_directories(gridtensor_test PRIVATE ${googletest_SOURCE_DIR}/googletest/include)
  target_include_directories(gridtensor_test PRIVATE ${googletest_SOURCE_DIR}/googlemock/include)
endif()

##
## Documentation (Doxygen)
##

if (BUILD_DOCS)
  find_package(Doxygen)

  if(DOXYGEN_FOUND)
    message("Generating documentation")
    set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/doxygen)
    set(DOXYFILE_IN ${CMAKE_SOURCE_DIR}/doxygen/Doxyfile.in)
    set(DOXYFILE_SOURCE_DIR "${CMAKE_SOURCE_DIR}/framework")
    include(UseDoxygen)
  else()
    message(FATAL_ERROR "Doxygen needs to be installed to generate documentation")
  endif()
endif()
