
CFLAGS 	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/mat/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c
EXAMPLESF       =
MANSEC          = Mat

include ${PETSC_DIR}/bmake/common/base

ex1: ex1.o  chkopts
	-${CLINKER}  -o ex1 ex1.o ${PETSC_MAT_LIB} 
	${RM} ex1.o

ex2: ex2.o  chkopts
	-${CLINKER}  -o ex2 ex2.o ${PETSC_MAT_LIB}
	${RM} ex2.o

ex3: ex3.o  chkopts
	-${CLINKER}  -o ex3 ex3.o ${PETSC_MAT_LIB}
	${RM} ex3.o

ex4: ex4.o  chkopts
	-${CLINKER}  -o ex4 ex4.o ${PETSC_MAT_LIB}
	${RM} ex4.o

ex5: ex5.o  chkopts
	-${CLINKER}  -o ex5 ex5.o ${PETSC_MAT_LIB}
	${RM} ex5.o

ex6: ex6.o  chkopts
	-${CLINKER}  -o ex6 ex6.o ${PETSC_MAT_LIB}
	${RM} ex6.o

runex1: 
	-@${MPIRUN} -np 1 ./ex1 -f0 ${DATAFILESPATH}/matrices/medium -viewer_binary_skip_info

runex3: 
	-@${MPIRUN} -np 1 ./ex3

runex5: 
	-@${MPIRUN} -np 2 ./ex5 -f ${DATAFILESPATH}/matrices/medium -viewer_binary_skip_info

#-------------------------------------------------------------
TESTEXAMPLES_C  = ex4.PETSc ex4.rm ex5.PETSc ex5.rm
TESTEXAMPLES_C_X11  = ex2.PETSc ex2.rm
TESTEXAMPLES_FORTRAN  = 
TESTEXAMPLES_C_X11_MPIUNI  =
TESTEXAMPLES_6  = ex1.PETSc runex1 ex1.rm ex5.PETSc runex5 ex5.rm
TESTEXAMPLES_FORTRAN_MPIUNI  =
TESTEXAMPLES_13 = ex1.PETSc ex1.rm


include ${PETSC_DIR}/bmake/common/test

