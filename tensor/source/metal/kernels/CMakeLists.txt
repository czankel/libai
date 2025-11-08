# Copyright (C) Chris Zankel. All rights reserved.
# This code is subject to U.S. and other copyright laws and
# intellectual property protections.
#
# The contents of this file are confidential and proprietary to Chris Zankel.

set_property(GLOBAL PROPERTY libai_metal_PATH ${CMAKE_CURRENT_BINARY_DIR})
set(libai_metal_PATH ${CMAKE_CURRENT_BINARY_DIR})

set(
  KERNELS
  "binary"
  "gemm"
  "rms_norm"
  "softmax"
  "unary"
)

function(build_kernel TARGET SRCFILE)
  message("BUILD ${TARGET}.air ${SRCFILE}")
  add_custom_command(
    COMMAND xcrun -sdk macosx metal -Wall -Wextra
                  -fno-fast-math
                  -c ${SRCFILE}
                  -I${PROJECT_SOURCE_DIR}
                  -o ${TARGET}.air
    DEPENDS ${SRCFILE} ## FIXME ${HEADERS}
    OUTPUT ${TARGET}.air
    COMMENT "Building ${TARGET}.air"
    VERBATIM
  )
endfunction(build_kernel)

foreach(KERNEL ${KERNELS})
  set(SRCFILE ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL}.metal)
  build_kernel(${KERNEL} ${SRCFILE})
  set(KERNELS_AIR ${KERNEL}.air ${KERNELS_AIR})
endforeach()

add_custom_command(
  OUTPUT ${libai_metal_PATH}/libai.metallib
  COMMAND xcrun -sdk macosx metallib ${KERNELS_AIR} -o ${libai_metal_PATH}/libai.metallib
  DEPENDS ${KERNELS_AIR}
  COMMENT "Building libai.metallib"
  VERBATIM
)

add_custom_target(
  libai_metallib
  DEPENDS ${libai_metal_PATH}/libai.metallib
)
