
ALL: lib

LIBBASE  = libpetscksp
DIRS     = jacobi none sor shell bjacobi mg eisens asm ksp composite redundant spai is pbjacobi ml\
           mat hypre tfs fieldsplit factor prometheus galerkin openmp supportgraph asa cp wb python ainvcusp sacusp bicgstabcusp\
           lsc redistribute gasm svd gamg parms bddc
LOCDIR   = src/ksp/pc/impls/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

