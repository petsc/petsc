# $Id: makefile,v 1.11 1998/06/13 05:09:47 balay Exp bsmith $ 

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

ex1: ex1.o   chkopts 
	-${CLINKER} -o ex1 ex1.o ${SLES_LIB}
	${RM} ex1.o

