
ALL: lib

CFLAGS   =  ${PNETCDF_INCLUDE} ${MATLAB_INCLUDE}
FFLAGS   =
SOURCEC  = bvec2.c bvec1.c dvec2.c vseqcr.c
SOURCEF  =
SOURCEH  = 
OBJSC    = bvec2.o  bvec1.o dvec2.o vseqcr.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/seq/
DIRS     = ftn-kernels

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

