
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/vec/is/is/examples/tests/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c
EXAMPLESF       = ex1f.F90 ex2f.F90

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

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

ex6: ex6.o chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_VEC_LIB}
	${RM} -f ex6.o

ex7: ex7.o chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_VEC_LIB}
	${RM} -f ex7.o

ex8: ex8.o chkopts
	-${CLINKER} -o ex8 ex8.o  ${PETSC_VEC_LIB}
	${RM} -f ex8.o

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

runex6_3:
	-@${MPIEXEC} -n 3 ./ex6 > ex6_3.tmp 2>&1;                                                 \
	  ${DIFF} output/ex6_3.out ex6_3.tmp || printf "${PWD}\nPossible problems with ex6_3, diffs above\n=========================================\n"; \
	  ${RM} -f ex6_3.tmp

runex7:
	-@${MPIEXEC} -n 1  ./ex7

runex8:
	-@${MPIEXEC} -n 1  ./ex8 > ex8_1.tmp 2>&1; \
	  ${DIFF} output/ex8_1.out ex8_1.tmp || printf "${PWD}\nPossible problems with ex8_1, diffs above\n=========================================\n"; \
	  ${RM} -f ex8_1.tmp

TESTEXAMPLES_C		    = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex5.PETSc runex5 ex5.rm ex6.PETSc runex6_3 ex6.rm ex7.PETSc runex7 ex7.rm ex8.PETSc runex8 ex8.rm
TESTEXAMPLES_C_X	    =
TESTEXAMPLES_FORTRAN	    = ex2f.PETSc runex2f ex2f.rm
TESTEXAMPLES_F90	    = ex1f.PETSc runex1f ex1f.rm
TESTEXAMPLES_FORTRAN_MPIUNI =
TESTEXAMPLES_C_X_MPIUNI   =
TESTEXAMPLES_13		    = ex3.PETSc ex3.rm ex4.PETSc ex4.rm

include ${PETSC_DIR}/lib/petsc/conf/test
