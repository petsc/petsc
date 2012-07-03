#requirespackage 'PETSC_HAVE_CUSP'
ALL: lib

CFLAGS    =
FFLAGS    =
SOURCECU  = aijcusp.cu aijAssemble.cu
SOURCEF   =
SOURCEH   = cuspmatimpl.h
OBJSCU    = aijcusp.o  aijAssemble.o
OBJSF     =
LIBBASE   = libpetscmat
DIRS      =
MANSEC    = Mat
LOCDIR    = src/mat/impls/aij/seq/seqcusp/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
