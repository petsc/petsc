-include ../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_HYPRE'

CFLAGS   = ${HYPRE_INCLUDE}
SOURCEC  = mhypre.c
LIBBASE  = libpetscmat
MANSEC   = Mat

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
