
CFLAGS	        = 
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ksp/pc/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c
EXAMPLESF       =
MANSEC          = PC

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER}  -o ex1 ex1.o ${PETSC_KSP_LIB} 
	${RM} ex1.o

ex2: ex2.o  chkopts
	-${CLINKER}  -o ex2 ex2.o ${PETSC_KSP_LIB} 
	${RM} ex2.o

#------------------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 -pc_factor_shift_type positive_definite 
	-@${RM} -f ex1.tmp

TESTEXAMPLES_C	   = ex1.PETSc runex1 ex1.rm 
TESTEXAMPLES_C_X = ex2.PETSc ex2.rm

include ${PETSC_DIR}/conf/test

