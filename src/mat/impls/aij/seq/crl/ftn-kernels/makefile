-include ../../../../../../../petscdir.mk
# This directory contains some computational kernels written in
# Fortran77.  These are mainly intended to provide high performance for
# COMPLEX arithmetic (since most C++ compilers do not do a good job
# optimizing complex number calculations), but can also be used to
# compare against C versions of the same kernels.
#requiresdefine   'PETSC_USE_FORTRAN_KERNELS'

SOURCEC  =
SOURCEF90= fmultcrl.F90
SOURCEH  =
LIBBASE  = libpetscsys
MANSEC   = Sys
LOCDIR   = src/mat/impls/aij/seq/crl/ftn-kernels/
DIRS     =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test


