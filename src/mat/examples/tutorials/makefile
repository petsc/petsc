
CFLAGS 	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/mat/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c ex4.c ex5.c ex6.c ex8.c ex9.c
EXAMPLESF       =
MANSEC          = Mat

include ${PETSC_DIR}/conf/base

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

ex6: ex6.o  chkopts
	-${CLINKER}  -o ex6 ex6.o ${PETSC_MAT_LIB}
	${RM} ex6.o

ex8: ex8.o  chkopts
	-${CLINKER}  -o ex8 ex8.o ${PETSC_MAT_LIB}
	${RM} ex8.o
ex9: ex9.o  chkopts
	-${CLINKER}  -o ex9 ex9.o ${PETSC_MAT_LIB}
	${RM} ex9.o
# See http://www.mcs.anl.gov/petsc/petsc-as/documentation/faq.html#datafiles for how to obtain the datafiles used below
runex1: 
	-@${MPIEXEC} -n 1 ./ex1 -f0 ${DATAFILESPATH}/matrices/medium -viewer_binary_skip_info

runex5: 
	-@${MPIEXEC} -n 2 ./ex5 -f ${DATAFILESPATH}/matrices/medium -viewer_binary_skip_info

runex9: 
	-@${MPIEXEC} -n 2 ./ex9 -f ${DATAFILESPATH}/matrices/medium -viewer_binary_skip_info

#-------------------------------------------------------------
TESTEXAMPLES_C		    = ex4.PETSc ex4.rm ex5.PETSc ex5.rm
TESTEXAMPLES_C_X11	    = ex2.PETSc ex2.rm ex9.PETSc runex9 ex9.rm
TESTEXAMPLES_FORTRAN	    = 
TESTEXAMPLES_C_X11_MPIUNI   = 
TESTEXAMPLES_C_NOCOMPLEX    = ex1.PETSc runex1 ex1.rm ex5.PETSc runex5 ex5.rm
TESTEXAMPLES_FORTRAN_MPIUNI = 
TESTEXAMPLES_13		    = ex1.PETSc ex1.rm


include ${PETSC_DIR}/conf/test

