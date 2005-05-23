
#requirespackage  'PETSC_HAVE_HYPRE'
#requiresprecision double
#requiresscalar    real

ALL: lib

CFLAGS   = ${HYPRE_INCLUDE}
FFLAGS   =
SOURCEC  = vhyp.c
SOURCEF  =
SOURCEH  = 
OBJSC    = vhyp.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/hypre/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

