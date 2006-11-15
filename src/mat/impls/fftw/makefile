
#requirespackage   'PETSC_HAVE_FFTW'
#requiresprecision double
#requiresscalar    complex

ALL: lib

CFLAGS   = ${FFTW_INCLUDE}
FFLAGS   =
SOURCEC  = fftw.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = fftw.o 
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = Mat
LOCDIR   = src/mat/impls/fftw/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
