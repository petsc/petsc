-include ../petscdir.mk
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = mffd.c  mffddef.c mfregis.c wp.c
SOURCEF  =
SOURCEH  = mffdimpl.h
LIBBASE  = libpetscmat
MANSEC   = Mat
DIRS     =
LOCDIR   = src/mat/impls/mffd/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

