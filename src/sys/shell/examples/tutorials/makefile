
CFLAGS 	        = 
FFLAGS	        = 
CPPFLAGS        = 
FPPFLAGS        =
LOCDIR          = src/sys/shell/examples/tutorials/
EXAMPLESC       = ex1.c
EXAMPLESF       =
MANSEC          = Sys

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

# Requires Python and petsc4py
ex1: ex1.o chkopts
	-${CLINKER}  -o ex1 ex1.o ${PETSC_LIB} 
	${RM} ex1.o

# See http://www.mcs.anl.gov/petsc/documentation/faq.html#datafiles for how to obtain the datafiles used below
runex1: 
	-@${MPIEXEC} -n 1 ./ex1 

#-------------------------------------------------------------
TESTEXAMPLES_C		    = ex1.PETSc 
TESTEXAMPLES_C_X	    = 
TESTEXAMPLES_FORTRAN	    = 
TESTEXAMPLES_C_X_MPIUNI   = 
TESTEXAMPLES_DATAFILESPATH  = 
TESTEXAMPLES_FORTRAN_MPIUNI = 
TESTEXAMPLES_13		    = 
TESTEXAMPLES_PARAMETIS      = 


include ${PETSC_DIR}/conf/test

