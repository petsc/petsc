-include ../../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_HARA'

ALL: lib

FFLAGS    =
SOURCECXX = pcharacpu.cxx
SOURCECU  = pchara.cu
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscksp
DIRS      =
MANSEC    = PC
LOCDIR    = src/ksp/pc/impls/hara/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
