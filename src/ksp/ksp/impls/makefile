-include ../petscdir.mk
ALL: lib

LIBBASE  = libpetscksp
DIRS     = cr bcgs bcgsl cg cgs gmres cheby rich lsqr preonly tcqmr tfqmr \
           qcg bicg minres symmlq lcd ibcgs python gcr fcg tsirm fetidp hpddm
LOCDIR   = src/ksp/ksp/impls/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
