
# This directory contains some computational kernels written in
# Fortran77.  These are mainly intended to provide high performance for
# COMPLEX arithmetic (since most C++ compilers do not do a good job
# optimizing complex number calculations), but can also be used to
# compare against C versions of the same kernels.

ALL: lib
#requiresdefine   'PETSC_USE_FORTRAN_KERNELS'

CFLAGS   =
FFLAGS   =
SOURCEC  =
SOURCEF90= fwaxpy.F90 faypx.F90 fnorm.F90 fxtimesy.F90 fmdot.F90 fmaxpy.F90
SOURCEH  =
LIBBASE  = libpetscvec
MANSEC   = Sys
LOCDIR   = src/vec/vec/impls/seq/ftn-kernels/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test


