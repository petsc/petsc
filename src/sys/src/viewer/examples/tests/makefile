# $Id: makefile,v 1.16 1999/02/01 22:18:56 balay Exp balay $ 

ALL: ex1

CFLAGS	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sys/src/viewer/examples/tests/

TESTEXAMPLES_2  = 
TESTEXAMPLES_4  = ex3.PETSc ex3.rm clean
BUILDEXAMPLES_4 = ex3.PETSc ex3.rm clean

EXAMPLESC       = ex3.c

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex3: ex3.o   chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_SYS_LIB}
	${RM} -f ex3.o

