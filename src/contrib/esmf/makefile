# $Id: makefile,v 1.2 2000/09/06 12:30:02 bsmith Exp bsmith $ 

ALL: lib

CFLAGS     =
FFLAGS     =
SOURCEC    = cesmfgrid.c
OBJSC      = cesmfgrid.o
SOURCEF    = esmf.F90 esmfgrid.F90 esmffield.F90
OBJSF      = esmf.o esmfgrid.o esmffield.o 
SOURCEH    = esmf.h esmfgrid.h
EXAMPLESC  = ex1.c
EXAMPLESF  = ex1f90.F90
LIBBASE    = libpetscfortran
LOCDIR     = src/contrib/esmf
TESTS      = 

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1f90: ex1.o ex1f90.o chkopts
	-${FLINKER} -o ex1f90 ex1f90.o ex1.o ${PETSC_FORTRAN_LIB} ${PETSC_LIB}
	${RM} -f ex1f90.o ex1.o

include ${PETSC_DIR}/bmake/common_test
