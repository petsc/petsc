# $Id: makefile,v 1.5 2001/06/02 03:10:55 bsmith Exp bsmith $ 



#
CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/pf/examples/tutorials/
EXAMPLESC	 = ex1.c
EXAMPLESF	 = 

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: ex1.o  chkopts  
	-${CLINKER} -o ex1 ex1.o   ${PETSC_LIB}
	${RM} -f ex1.o 
#--------------------------------------------------------------------------
TESTEXAMPLES_1	 = ex1.PETSc ex1.rm
TESTEXAMPLES_2	 = 
TESTEXAMPLES_3	 = 
TESTEXAMPLES_4	 = 
TESTEXAMPLES_9	 = 
TESTEXAMPLES_12	 = 
TESTEXAMPLES_13  = 

include ${PETSC_DIR}/bmake/common_test
