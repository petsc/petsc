
ALL: lib

LIBBASE  = libpetscksp
DIRS     = jacobi none sor shell bjacobi mg eisens asm ksp composite redundant spai is pbjacobi ml\
           mat hypre samg tfs fieldsplit factor prometheus galerkin openmp
LOCDIR   = src/ksp/pc/impls/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

