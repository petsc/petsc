# $Id: makefile,v 1.23 1999/07/26 18:47:48 curfman Exp bsmith $ 

ALL: ex1

CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/

TESTEXAMPLES_1  = ex1.PETSc runex1 ex1.rm
TESTEXAMPLES_2  = 
TESTEXAMPLES_3  = ex1f.PETSc runex1f ex1f.rm
TESTEXAMPLES_4  =  
TESTEXAMPLES_13 = ex2.PETSc ex2.rm ex3.PETSc ex3.rm ex4.PETSc ex4.rm \
		  ex5.PETSc ex5.rm clean
BUILDEXAMPLES_1 = ex2.PETSc ex4.PETSc
BUILDEXAMPLES_3 = ex2f.PETSc 
BUILDEXAMPLES_4 = 

EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c util2.c
EXAMPLESF       = ex1f.F ex2f.F ex2f.h

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_TS_LIB}
	${RM} ex1.o

ex1f: ex1f.o  chkopts
	-${FLINKER} -o ex1f ex1f.o ${PETSC_FORTRAN_LIB} ${PETSC_TS_LIB}
	${RM} ex1f.o

ex2: ex2.o util2.o chkopts
	-${CLINKER} -o ex2 ex2.o util2.o ${PETSC_TS_LIB}
	${RM} ex2.o util2.o

ex2f: ex2f.o util2.o chkopts
	-${FLINKER} -o ex2f ex2f.o util2.o ${PETSC_FORTRAN_LIB} ${PETSC_TS_LIB}
	${RM} ex2f.o util2.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_TS_LIB}
	${RM} ex3.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_TS_LIB}
	${RM} ex4.o

ex5: ex5.o  chkopts
	-${CLINKER} -o ex5 ex5.o  ${PETSC_TS_LIB}
	${RM} ex5.o

ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_TS_LIB}
	${RM} ex6.o

runex1:
	-@${MPIRUN} -np 1 ex1 -ksp_gmres_irorthog  > ex1_1.tmp 2>&1;	  \
	   if (diff output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   ${RM} -f ex1_1.tmp

runex1f:
	-@${MPIRUN} -np 1 ex1f -ksp_gmres_irorthog > ex1f_1.tmp 2>&1;	  \
	   if (diff output/ex1f_1.out ex1f_1.tmp) then true; \
	   else echo "Possible problem with ex1f_1, diffs above"; fi; \
	   ${RM} -f ex1f_1.tmp

