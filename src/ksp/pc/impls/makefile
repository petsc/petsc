# $Id: makefile,v 1.40 2001/08/22 18:01:21 balay Exp $

ALL: lib

LIBBASE  = libpetscksp
DIRS     = jacobi none sor lu shell bjacobi mg eisens ilu icc asm ksp composite redundant spai is cholesky pbjacobi \
           esi mat hypre ramg samg
LOCDIR   = src/ksp/pc/impls/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

