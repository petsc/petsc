# $Id: makefile,v 1.10 1999/02/02 21:36:15 balay Exp bsmith $ 

ALL: lib

CFLAGS     =
FFLAGS     =
EXAMPLESC  = test_IRIX.c
EXAMPLESF  = test_IRIXf90.F
OBJSF      = 
LIBBASE    = libpetscfortran
LOCDIR     = src/fortran/f90/tests
TESTS      = test_IRIX
CLEANFILES =
include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test

test_IRIX: test_IRIX.o test_IRIXf90.o
	-${FLINKER} -o test_IRIX test_IRIX.o test_IRIXf90.o
	${RM} -f test_IRIX.o test_IRIXf90.o
