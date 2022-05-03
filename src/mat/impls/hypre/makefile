-include ../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_HYPRE'

CFLAGS   = ${HYPRE_INCLUDE}
SOURCEC  = mhypre.c
SOURCEF  =
SOURCEH  = mhypre.h
LIBBASE  = libpetscmat
DIRS     =
MANSEC   = Mat
LOCDIR   = src/mat/impls/hypre/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
