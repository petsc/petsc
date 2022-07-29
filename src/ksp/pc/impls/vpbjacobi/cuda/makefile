-include ../../../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_CUDA'

SOURCEC   =
SOURCECU  = vpbjacobi_cuda.cu
SOURCEH   =
LIBBASE   = libpetscksp
DIRS      =
MANSEC    = KSP
SUBMANSEC = PC
LOCDIR    = src/ksp/pc/impls/vpbjacobi/cuda

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
