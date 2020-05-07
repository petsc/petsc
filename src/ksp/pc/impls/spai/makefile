-include ../../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_SPAI'

ALL: lib
#
#
CFLAGS    =
FFLAGS    =
CPPFLAGS  = ${SPAI_INCLUDE}
SOURCEC   = ispai.c dspai.c
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscksp
DIRS      =
MANSEC    = KSP
SUBMANSEC = PC
LOCDIR    = src/ksp/pc/impls/spai/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test


