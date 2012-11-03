
#requirespackage   'PETSC_HAVE_UMFPACK'
#requiresprecision double

ALL: lib

CFLAGS   = ${UMFPACK_INCLUDE}
FFLAGS   =
SOURCEC  = umfpack.c
SOURCEF  = 
SOURCEH  = 
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/umfpack/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
