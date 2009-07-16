
#requirespackage  'PETSC_HAVE_HYPRE'
#requiresprecision double
#requiresscalar    real


ALL: lib

CFLAGS   = ${HYPRE_INCLUDE}
FFLAGS   =
SOURCEC  = mhyp.c
SOURCEF  =
SOURCEH  = 
OBJSC    = mhyp.o
OBJSF    =
LIBBASE  = libpetscdm
MANSEC   = Mat
LOCDIR   = src/mat/impls/hypre/

include ${PETSC_DIR}/conf/base
include ${PETSC_DIR}/conf/test

