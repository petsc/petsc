# $Id: makefile,v 1.10 1998/04/23 22:46:41 balay Exp balay $ 

ALL: ex1

CFLAGS	        = 
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ec/examples/tutorials/

TESTEXAMPLES_1  =
TESTEXAMPLES_2  =
TESTEXAMPLES_4  =
BUILDEXAMPLES_1 =
BUILDEXAMPLES_2 =
BUILDEXAMPLES_4 =

EXAMPLESC       = 

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: ex1.o 
	-${CLINKER} -o ex1 ex1.o ${SLES_LIB}
	${RM} ex1.o

