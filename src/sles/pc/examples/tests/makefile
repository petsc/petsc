# $Id: makefile,v 1.62 1999/02/01 21:25:47 balay Exp bsmith $ 

ALL: ex1

CFLAGS	        =
FFLAGS          =
CPPFLAGS        = ${BS_INCLUDE}
FPPFLAGS        =
LOCDIR          = src/sles/pc/examples/tests/

TESTEXAMPLES_1  = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex3.PETSc \
		  runex3 ex3.rm ex4.PETSc runex4 ex4.rm  clean
TESTEXAMPLES_2  = ex5.PETSc runex5 ex5.rm clean
TESTEXAMPLES_4  = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex3.PETSc \
		  runex3 ex3.rm ex4.PETSc runex4 ex4.rm ex5.PETSc runex5 ex5.rm clean
BUILDEXAMPLES_1 = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex3.PETSc \
		  ex3.rm ex4.PETSc ex4.rm  ex6.PETSc ex6.rm ex7.PETSc ex7.rm clean
BUILDEXAMPLES_2 = ex5.PETSc ex5.rm clean
BUILDEXAMPLES_4 = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex3.PETSc \
		  ex3.rm ex4.PETSc ex4.rm ex5.PETSc ex5.rm clean

EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c
DOCS            = readme

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: ex1.o   chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_LIB}
	${RM} ex1.o

ex2: ex2.o   chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_LIB}
	${RM} ex2.o

ex3: ex3.o   chkopts
	-${CLINKER} -o ex3 ex3.o ${PETSC_LIB}
	${RM} ex3.o

ex4: ex4.o   chkopts
	-${CLINKER} -o ex4 ex4.o ${PETSC_LIB}
	${RM} ex4.o

ex5: ex5.o   chkopts
	-${CLINKER} -o ex5 ex5.o ${PETSC_LIB}
	${RM} ex5.o

ex6: ex6.o   chkopts
	-${CLINKER} -o ex6 ex6.o ${PETSC_LIB}
	${RM} ex6.o

ex7: ex7.o   chkopts
	-${CLINKER} -o ex7 ex7.o ${PETSC_LIB}
	${RM} ex7.o

runex1:
	-@${MPIRUN} -np 1 ex1 > ex1_1.tmp 2>&1;   \
	   if (diff output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   ${RM} -f ex1_1.tmp 
runex2:
	-@${MPIRUN} -np 1 ex2 -ksp_type cg -ksp_smonitor > ex2_1.tmp 2>&1;	 \
	   if (diff output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi; \
	   ${RM} -f ex2_1.tmp 
runex3:
	-@${MPIRUN} -np 1 ex3 -ksp_type gmres -ksp_smonitor -pc_type sor\
			     -pc_sor_symmetric > ex3_1.tmp 2>&1;   \
	   if (diff output/ex3_1.out ex3_1.tmp) then true; \
	   else echo "Possible problem with ex3_1, diffs above"; fi; \
	   ${RM} -f ex3_1.tmp 
runex4:
	-@${MPIRUN} -np 1 ex4	 > ex4_1.tmp 2>&1;   \
	   if (diff output/ex4_1.out ex4_1.tmp) then true; \
	   else echo "Possible problem with ex4_1, diffs above"; fi; \
	   ${RM} -f ex4_1.tmp 

runex5:
	-@${MPIRUN} -np 1 ex5 > ex5_1.tmp 2>&1;   \
	   if (diff output/ex5_1.out ex5_1.tmp) then true; \
	   else echo "Possible problem with ex5_1, diffs above"; fi; \
	   ${RM} -f ex5_1.tmp 

runex6:
	-@${MPIRUN} -np 1  ex6 -m 7 > ex6_1.tmp   2>&1;   \
	   if (diff output/ex6_1.out ex6_1.tmp) then true; \
	   else echo "Possible problem with ex6_1, diffs above"; fi; \
	   ${RM} -f ex6_1.tmp
