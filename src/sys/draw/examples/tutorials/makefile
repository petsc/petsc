
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        = 
LOCDIR          = src/sys/draw/examples/tutorials/
EXAMPLESC       =
EXAMPLESF       =
MANSEC          = Draw

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

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
runex1:
	-@${MPIEXEC} -n 1 ./ex1
runex2:
	-@${MPIEXEC} -n 1 ./ex2
runex3:
	-@${MPIEXEC} -n 1 ./ex3
runex4:
	-@${MPIEXEC} -n 1 ./ex4
runex4_2:
	-@${MPIEXEC} -n 2 ./ex4

TESTEXAMPLES_C_X	  = 
TESTEXAMPLES_FORTRAN	  = 
TESTEXAMPLES_C_X_MPIUNI = 

include ${PETSC_DIR}/conf/test
