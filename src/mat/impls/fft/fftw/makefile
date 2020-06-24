-include ../../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_FFTW'

ALL: lib

CFLAGS   = ${FFTW_INCLUDE}
FFLAGS   =
SOURCEC  = fftw.c
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscmat
MANSEC   = Mat
LOCDIR   = src/mat/impls/fft/fftw/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
