#
CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/vec/pf/examples/tutorials/
EXAMPLESC	 = 
EXAMPLESF	 = 
MANSEC           = Vec

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o  chkopts  
	-${CLINKER} -o ex1 ex1.o   ${PETSC_DM_LIB}
	${RM} -f ex1.o 
#--------------------------------------------------------------------------

TESTEXAMPLES_C		    = 
TESTEXAMPLES_C_X11	    = 
TESTEXAMPLES_FORTRAN	    = 
TESTEXAMPLES_C_X11_MPIUNI   = 
TESTEXAMPLES_FORTRAN_MPIUNI = 
TESTEXAMPLES_F90	    = 
TESTEXAMPLES_13		    = 

include ${PETSC_DIR}/conf/test
