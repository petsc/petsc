#requirespackage 'PETSC_HAVE_CUSP'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCECU = veccusp.cu
SOURCEF  =
SOURCEH  = cuspvecimpl.h
OBJSCU   = veccusp.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/seq/seqcusp/
DIRS     = 

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

