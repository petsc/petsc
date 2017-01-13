
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR   = src/sys/classes/draw/examples/tutorials/
EXAMPLESC       =
EXAMPLESF       =
MANSEC          = Sys
SUBMANSEC       = Draw

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_LIB}
	${RM} -f ex1.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_LIB}
	${RM} -f ex2.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_LIB}
	${RM} -f ex3.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_LIB}
	${RM} -f ex4.o

ex5: ex5.o  chkopts
	-${FLINKER} -o ex5 ex5.o  ${PETSC_LIB}
	${RM} -f ex5.o

ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_LIB}
	${RM} -f ex6.o

#------------------------------------------------------------------
TESTEXAMPLES_C_X	  =
TESTEXAMPLES_FORTRAN	  =
TESTEXAMPLES_C_X_MPIUNI =

include ${PETSC_DIR}/lib/petsc/conf/test
