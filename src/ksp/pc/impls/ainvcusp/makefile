#requiresdefine 'PETSC_HAVE_CUSP'
ALL: lib

CFLAGS    =
FFLAGS    =
SOURCECU  = ainvcusp.cu
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscksp
DIRS      =
MANSEC    = KSP
SUBMANSEC = PC
LOCDIR    = src/ksp/pc/impls/ainvcusp/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
