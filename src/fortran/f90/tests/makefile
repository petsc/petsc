# $Id: makefile,v 1.4 1998/03/25 17:36:38 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
FFLAGS   = 
SOURCEC  = test_IRIX.c
SOURCEF  = test_IRIXf90.F
SOURCEH  = 
OBJSC    = test_IRIX.o test_IRIXf90.o
OBJSF    = 
LIBBASE  = libpetscfortran
MANSEC   =
LOCDIR   = src/fortran/f90/tests
TESTS    = test_IRIX
include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base


test_IRIX: test_IRIX.o test_IRIXf90.o
	-${FLINKER} -o test_IRIX test_IRIX.o test_IRIXf90.o
	${RM} -f test_IRIX.o test_IRIXf90.o
