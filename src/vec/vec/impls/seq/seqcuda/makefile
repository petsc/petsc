#requirespackage 'PETSC_HAVE_CUDA'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = veccuda.c
SOURCECU = veccuda2.cu
SOURCEF  =
SOURCEH  = ../../../../../../include/petsc/private/cudavecimpl.h
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/seq/seqcuda/
DIRS     =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

