# $Id: makefile,v 1.5 2001/08/31 15:38:10 buschelm Exp $ 

#requireslanguage  CONLY
#requiresscalar    real 
#requiresprecision double

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = matmatmult.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = matmatmult.o
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = PC
LOCDIR   = src/contrib/lahaye
DIRS     = ramg samg

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

ex1: ex1.o matmatmult.o chkopts
	${CLINKER}  -o ex1 ex1.o matmatmult.o ${PETSC_MAT_LIB} 
ex2: ex2.o matmatmult.o chkopts
	${CLINKER}  -o ex2 ex2.o matmatmult.o ${PETSC_MAT_LIB} 
