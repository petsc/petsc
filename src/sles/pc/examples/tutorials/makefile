# $Id: makefile,v 1.14 1999/01/12 23:14:57 bsmith Exp bsmith $ 

ALL: ex1

CFLAGS	        = 
FFLAGS          =
CPPFLAGS        = ${BS_INCLUDE}
FPPFLAGS        =
LOCDIR          = src/sles/pc/examples/tutorials/

TESTEXAMPLES_1  =
TESTEXAMPLES_2  =
TESTEXAMPLES_4  =
BUILDEXAMPLES_1 =
BUILDEXAMPLES_2 =
BUILDEXAMPLES_4 =

EXAMPLESC       = 

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: ex1.o   chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_LIB}
	${RM} ex1.o

