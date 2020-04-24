-include ../petscdir.mk
#requiresdefine   'PETSC_HAVE_FORTRAN'
ALL: lib
CFLAGS   =
FFLAGS   =
SOURCEC  = zfftwf.c
SOURCEF  =
SOURCEH  =
DIRS     =
LIBBASE  = libpetscmat
LOCDIR   = src/mat/impls/fft/fftw/ftn-custom/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
