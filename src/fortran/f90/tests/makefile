# $Id: makefile,v 1.8 1998/11/05 17:22:26 balay Exp balay $ 

ALL: lib

CFLAGS     =
FFLAGS     =
EXAMPLESC  = test_IRIX.c
EXAMPLESF  = test_IRIXf90.F
OBJSF      = 
LIBBASE    = libpetscfortran
MANSEC     =
LOCDIR     = src/fortran/f90/tests
TESTS      = test_IRIX
CLEANFILES =
include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base


test_IRIX: test_IRIX.o test_IRIXf90.o
	-${FLINKER} -o test_IRIX test_IRIX.o test_IRIXf90.o
	${RM} -f test_IRIX.o test_IRIXf90.o
