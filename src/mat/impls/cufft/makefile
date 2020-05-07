-include ../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_CUDA'
#requiresprecision single
#requiresscalar    complex
ALL: lib

CFLAGS    =
FFLAGS    =
SOURCECU  = cufft.cu
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscmat
DIRS      =
MANSEC    = Mat
LOCDIR    = src/mat/impls/cufft/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
