# $Id: makefile,v 1.11 2001/07/23 02:52:23 bsmith Exp balay $ 
#
#   SPAI_LIB should be defined in bmake/$PETSC_ARCH/packages
#
ALL: ex1

CFLAGS 	        =
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/contrib/pc/spai/examples/tutorials/


include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

ex1: ex1.o  chkopts
	-${CLINKER}  -o ex1 ex1.o ${PETSC_LIB} ${SPAI_LIB} ${PETSC_LIB} 
	${RM} ex1.o

ex2: ex2.o  chkopts
	-${CLINKER}  -o ex2 ex2.o ${PETSC_LIB} ${SPAI_LIB} ${PETSC_LIB} 
	${RM} ex2.o

TESTEXAMPLES_1  =
TESTEXAMPLES_2  =
TESTEXAMPLES_3  =
TESTEXAMPLES_4  =
TESTEXAMPLES_6  =
TESTEXAMPLES_7  =
TESTEXAMPLES_9  =
BUILDEXAMPLES_1 =
BUILDEXAMPLES_2 =
BUILDEXAMPLES_3 =
BUILDEXAMPLES_4 =
BUILDEXAMPLES_6 =
BUILDEXAMPLES_7 =
BUILDEXAMPLES_9 =

EXAMPLESC       =  ex2.c
EXAMPLESF       =


