
CFLAGS	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/vec/is/examples/tests/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c
EXAMPLESF       = ex1f.F ex2f.F

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_VEC_LIB}
	${RM} -f ex1.o

ex1f: ex1f.o chkopts
	-${FLINKER} -o ex1f ex1f.o ${PETSC_VEC_LIB}
	${RM} -f ex1f.o

ex2: ex2.o chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_VEC_LIB}
	${RM} -f ex2.o

ex2f: ex2f.o chkopts
	-${FLINKER} -o ex2f ex2f.o ${PETSC_VEC_LIB}
	${RM} -f ex2f.o

ex3: ex3.o chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_VEC_LIB}
	${RM} -f ex3.o

ex4: ex4.o chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_VEC_LIB}
	${RM} -f ex4.o

ex5: ex5.o chkopts
	-${CLINKER} -o ex5 ex5.o  ${PETSC_VEC_LIB}
	${RM} -f ex5.o

#-------------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1  ./ex1 

runex1f:
	-@${MPIEXEC} -n 1  ./ex1f 


runex2:
	-@${MPIEXEC} -n 1  ./ex2 

runex2f:
	-@${MPIEXEC} -n 1  ./ex2f 

runex5:
	-@${MPIEXEC} -n 3  ./ex5 

TESTEXAMPLES_C		    = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex5.PETSc runex5 ex5.rm 
TESTEXAMPLES_C_X11	    = 
TESTEXAMPLES_FORTRAN	    = ex1f.PETSc runex1f ex1f.rm ex2f.PETSc runex2f ex2f.rm
TESTEXAMPLES_FORTRAN_MPIUNI = 
TESTEXAMPLES_C_X11_MPIUNI   = 
TESTEXAMPLES_13		    = ex3.PETSc ex3.rm ex4.PETSc ex4.rm 

include ${PETSC_DIR}/conf/test
