-include ../../../../../petscdir.mk
ALL: lib

CFLAGS    =
FFLAGS    =
SOURCEC   = telescope.c telescope_dmda.c telescope_coarsedm.c
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscksp
DIRS      =
MANSEC    = KSP
SUBMANSEC = PC
LOCDIR    = src/ksp/pc/impls/telescope/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
