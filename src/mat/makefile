-include ../petscdir.mk
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  =
SOURCEF  =
SOURCEH  = ../../include/petsc/private/matimpl.h \
           ../../include/petscmat.h \
           ../../include/petsc/private/kernels/blockinvert.h \
           ../../include/petsc/private/kernels/blocktranspose.h \
           ../../include/petsc/private/kernels/blockmatmult.h \
           ../../include/petscmatelemental.h
LIBBASE  = libpetscmat
DIRS     = interface impls tests tutorials utils matfd partition coarsen order color f90-mod ftn-kernels
LOCDIR   = src/mat/
MANSEC   = Mat

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
