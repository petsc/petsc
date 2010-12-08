#requirespackage 'PETSC_HAVE_CUDA'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCECU = veccuda.cu
SOURCEF  =
SOURCEH  = cudavecimpl.h
OBJSC    = veccuda.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/seq/seqcuda/
DIRS     = 

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

