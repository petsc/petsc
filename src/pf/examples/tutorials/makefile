# $Id: makefile,v 1.7 2001/08/22 18:02:56 balay Exp $ 



#
CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/pf/examples/tutorials/
EXAMPLESC	 = ex1.c
EXAMPLESF	 = 
MANSEC           = Vec

include ${PETSC_DIR}/bmake/common/base

ex1: ex1.o  chkopts  
	-${CLINKER} -o ex1 ex1.o   ${PETSC_DM_LIB}
	${RM} -f ex1.o 
#--------------------------------------------------------------------------
TESTEXAMPLES_1	 = ex1.PETSc ex1.rm
TESTEXAMPLES_2	 = 
TESTEXAMPLES_3	 = 
TESTEXAMPLES_4	 = 
TESTEXAMPLES_9	 = 
TESTEXAMPLES_12	 = 
TESTEXAMPLES_13  = 

include ${PETSC_DIR}/bmake/common/test
