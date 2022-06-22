-include ../../../../../petsckdir.mk
#requirespackage  'PETSC_HAVE_HYPRE'

CFLAGS   = ${HYPRE_INCLUDE}
SOURCEC  = vhyp.c
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/hypre/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

