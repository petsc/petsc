-include ../petscdir.mk
ALL: lib

CFLAGS    =
FFLAGS    =
SOURCEC   = bipartite.c valid.c weights.c
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscmat
MANSEC    = Mat
SUBMANSEC = MatOrderings
LOCDIR    = src/mat/color/utils/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
