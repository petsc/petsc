#requiresdefine 'PETSC_HAVE_CUSP'
ALL: lib

CFLAGS    =
FFLAGS    =
SOURCECU  = bicgstabcusp.cu
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscksp
DIRS      =
MANSEC    = KSP
SUBMANSEC = PC
LOCDIR    = src/ksp/pc/impls/bicgstabcusp/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
