#requirespackage 'PETSC_HAVE_TXPETSCGPU'
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

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
