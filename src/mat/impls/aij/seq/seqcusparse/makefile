#requirespackage 'PETSC_HAVE_VECCUDA'
ALL: lib

CFLAGS    =
FFLAGS    =
SOURCECU  = aijcusparse.cu
SOURCEF   =
SOURCEH   = cusparsematimpl.h
LIBBASE   = libpetscmat
DIRS      =
MANSEC    = Mat
LOCDIR    = src/mat/impls/aij/seq/seqcusparse/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
