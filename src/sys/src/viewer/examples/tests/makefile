# $Id: makefile,v 1.9 1998/04/23 22:50:08 balay Exp balay $ 

ALL: ex1

CFLAGS	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/viewer/examples/tests/

TESTEXAMPLES_2  = 
TESTEXAMPLES_4  = ex1.PETSc ex1.rm ex2.PETSc ex2.rm clean
BUILDEXAMPLES_4 = ex1.PETSc ex1.rm ex2.PETSc ex2.rm clean

EXAMPLESC       = ex1.c ex2.c ex1.m

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

runex2:
	@echo "There are no externally runable examples in this directory"

ex1: ex1.o   chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_LIB}
	${RM} -f ex1.o

ex2: ex2.o   chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_LIB}
	${RM} -f ex2.o


