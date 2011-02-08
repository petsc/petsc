
#requirespackage   'PETSC_HAVE_FFTW'
#requiresprecision double

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
LOCDIR   = src/mat/impls/fft/fftw/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
