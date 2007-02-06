
#requirespackage  'PETSC_HAVE_HYPRE'
#requiresprecision double
#requiresscalar    real


ALL: lib

CFLAGS   = ${HYPRE_INCLUDE}
FFLAGS   =
SOURCEC  = hypre.c
SOURCEF  =
SOURCEH  = 
OBJSC    = hypre.o
OBJSF    =
LIBBASE  = libpetscksp
MANSEC   = PC
LOCDIR   = src/ksp/pc/impls/hypre/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

