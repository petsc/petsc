
CFLAGS 	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/mat/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c ex4.c
EXAMPLESF       =
MANSEC          = Mat

include ${PETSC_DIR}/bmake/common/base

ex1: ex1.o  chkopts
	-${CLINKER}  -o ex1 ex1.o ${PETSC_MAT_LIB} 
	${RM} ex1.o

ex2: ex2.o  chkopts
	-${CLINKER}  -o ex2 ex2.o ${PETSC_MAT_LIB}
	${RM} ex2.o

ex4: ex4.o  chkopts
	-${CLINKER}  -o ex4 ex4.o ${PETSC_MAT_LIB}
	${RM} ex4.o

ex5: ex5.o  chkopts
	-${CLINKER}  -o ex5 ex5.o ${PETSC_MAT_LIB}
	${RM} ex5.o

runex1: 
	-@${MPIRUN} ex1 -f0 ${DATAFILESPATH}/matrices/medium -load_ignore_info

runex5: 
	-@${MPIRUN} -np 2 ex5 -f ${DATAFILESPATH}/matrices/medium -load_ignore_info

#-------------------------------------------------------------
TESTEXAMPLES_1  = ex4.PETSc ex4.rm ex5.PETSc runex5 ex5.rm
TESTEXAMPLES_2  = ex2.PETSc ex2.rm
TESTEXAMPLES_3  = 
TESTEXAMPLES_4  =
TESTEXAMPLES_6  = ex1.PETSc runex1 ex1.rm 
TESTEXAMPLES_9  =
TESTEXAMPLES_13 = ex1.PETSc ex1.rm


include ${PETSC_DIR}/bmake/common/test

