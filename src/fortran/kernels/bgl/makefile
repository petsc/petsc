
# This directory contains some computational kernels written in
# Fortran77.  These are mainly intended to provide high performance for
# COMPLEX arithmetic (since most C++ compilers do not do a good job
# optimizing complex number calculations), but can also be used to
# compare against C versions of the same kernels.

ALL: lib
SPECIALFASTLIB = yes
specialfastlib: lib
#requirespackage   'PETSC_USE_BGL_KERNELS'

CFLAGS   =
FFLAGS   =
SOURCEC  = bgl_align.c
SOURCEF  = fcopy_bgl.F fmdot_bgl.F fsolvebaij_bgl.F sgemv_bgl.F fmaxpy_bgl.F fnorm_bgl.F fwaxpy_bgl.F

SOURCEH  = 
OBJSC    = ${SOURCEC:.c=.o}
OBJSF    = ${SOURCEF:.F=.o}
LIBBASE  = libpetsc
MANSEC   = Sys
LOCDIR   = src/fortran/kernels/bgl

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


