# $Id: makefile,v 1.9 1999/02/01 21:45:25 balay Exp balay $ 

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


test_IRIX: test_IRIX.o test_IRIXf90.o
	-${FLINKER} -o test_IRIX test_IRIX.o test_IRIXf90.o
	${RM} -f test_IRIX.o test_IRIXf90.o
