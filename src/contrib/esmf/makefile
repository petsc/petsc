# $Id: makefile,v 1.4 2000/09/13 03:13:53 bsmith Exp bsmith $ 

ALL: lib

#requirespackage  'PETSC_HAVE_F90_C'

CFLAGS     =
FFLAGS     =
SOURCEC    = cesmfgrid.c
OBJSC      = cesmfgrid.o
SOURCEF    = esmf.F esmfgrid.F esmffield.F
OBJSF      = esmf.o esmfgrid.o esmffield.o 
SOURCEH    = esmf.h esmfgrid.h
EXAMPLESC  = ex1.c
EXAMPLESF  = ex1F.F
LIBBASE    = libpetscfortran
LOCDIR     = src/contrib/esmf
TESTS      = 

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1f90: ex1.o ex1f90.o chkopts
	-${FLINKER} -o ex1f90 ex1f90.o ex1.o ${PETSC_FORTRAN_LIB} ${PETSC_LIB}
	${RM} -f ex1f90.o ex1.o

include ${PETSC_DIR}/bmake/common_test
