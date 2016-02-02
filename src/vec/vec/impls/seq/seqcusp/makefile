#requirespackage 'PETSC_HAVE_CUSP'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = veccusp.c
SOURCECU = veccusp2.cu vecscattercusp.cu
SOURCEF  =
SOURCEH  = cuspvecimpl.h
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/seq/seqcusp/
DIRS     =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

