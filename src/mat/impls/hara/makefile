-include ../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_HARA'

ALL: lib

FFLAGS    =
SOURCECXX = matharacpu.cxx
SOURCECU  = mathara.cu
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscmat
DIRS      =
MANSEC    = Mat
LOCDIR    = src/mat/impls/hara/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
