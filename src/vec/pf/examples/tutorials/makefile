#
CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/vec/pf/examples/tutorials/
EXAMPLESC	 = 
EXAMPLESF	 = 
MANSEC           = Vec

include ${PETSC_DIR}/bmake/common/base

ex1: ex1.o  chkopts  
	-${CLINKER} -o ex1 ex1.o   ${PETSC_DM_LIB}
	${RM} -f ex1.o 
#--------------------------------------------------------------------------
TESTEXAMPLES_1	 = 
TESTEXAMPLES_2	 = 
TESTEXAMPLES_3	 = 
TESTEXAMPLES_4	 = 
TESTEXAMPLES_9	 = 
TESTEXAMPLES_12	 = 
TESTEXAMPLES_13  = 

include ${PETSC_DIR}/bmake/common/test
