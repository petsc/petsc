# -*- makefile -*-

CFLAGS   = -I/u/dalcinl/Devel/Python/petsc4py/petsc/lib/ext -I/u/dalcinl/Devel/Python/petsc4py/petsc/lib/ext/include

EXAMPLESC = maij.c schur.c

include ${PETSC_DIR}/bmake/common/base


maij: maij.o schur.o chkopts
	-${CLINKER} -o maij maij.o schur.o ${PETSC_KSP_LIB}
	${RM} maij.o

include ${PETSC_DIR}/bmake/common/test
