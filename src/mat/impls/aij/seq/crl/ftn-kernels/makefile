
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
SOURCEF  = fmultcrl.F
SOURCEH  = 
LIBBASE  = libpetscsys
MANSEC   = Sys
LOCDIR   = src/mat/impls/aij/seq/crl/ftn-kernels/
DIRS     = 

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test


