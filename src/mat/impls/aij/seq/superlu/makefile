#requirespackage   'PETSC_HAVE_SUPERLU'
#requiresprecision double

ALL: lib

CFLAGS   = ${SUPERLU_INCLUDE}
FFLAGS   =
SOURCEC  = superlu.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = superlu.o
OBJSF    = 
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/superlu/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
