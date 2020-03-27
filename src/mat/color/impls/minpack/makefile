-include ../petscdir.mk
ALL: lib

CFLAGS    =
FFLAGS    =
SOURCEC   = color.c degr.c dsm.c ido.c numsrt.c seq.c setr.c slo.c
SOURCEF   =
SOURCEH   = color.h
LIBBASE   = libpetscmat
MANSEC    = Mat
SUBMANSEC = MatOrderings
LOCDIR    = src/mat/color/impls/minpack/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
