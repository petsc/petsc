-include ../../../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_CUDA'

CXXFLAGS = ${HPDDM_INCLUDE}
SOURCECU = hpddm.cu
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscksp
LOCDIR   = src/ksp/ksp/impls/hpddm/cuda/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
