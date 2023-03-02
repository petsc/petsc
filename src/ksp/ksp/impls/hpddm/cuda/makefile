-include ../../../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_CUDA'

CXXFLAGS = ${HPDDM_INCLUDE}
SOURCECU = hpddm.cu
LIBBASE  = libpetscksp

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
