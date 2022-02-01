-include ../../../../../petscdir.mk
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = bcgs.c
SOURCEF  =
LIBBASE  = libpetscksp
DIRS     = fbcgs fbcgsr qmrcgs pipebcgs
MANSEC   = KSP
LOCDIR   = src/ksp/ksp/impls/bcgs/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
