# $Id: makefile,v 1.66 1999/07/08 14:06:26 bsmith Exp bsmith $ 

ALL: ex1

CFLAGS          = 
FFLAGS          = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/snes/examples/tests/umin/

TESTEXAMPLES_1  = ex2.PETSc runex2 runex2_2 runex2_3 ex2.rm ex3.PETSc\
		  runex3 ex3.rm clean
TESTEXAMPLES_2  = ex3.PETSc runex3_2 ex3.rm clean
TESTEXAMPLES_8  = 
TESTEXAMPLES_13 = 
BUILDEXAMPLES_1 = ex2.PETSc ex2.rm ex3.PETSc ex3.rm clean
BUILDEXAMPLES_2 = ex3.PETSc ex3.rm clean
BUILDEXAMPLES_8 = 

EXAMPLESC       = ex2.c ex3.c
EXAMPLESF       = 
DOCS            = readme
DIRS            = 

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex2: ex2.o chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_LIB}
	${RM} ex2.o

ex3: ex3.o chkopts
	-${CLINKER} -o ex3 ex3.o ${PETSC_LIB}
	${RM} ex3.o

runex2:
	-@${MPIRUN} -np 1 ex2 -p 1 -snes_smonitor > ex2_1.tmp 2>&1;   \
           if (diff output/ex2_1.out ex2_1.tmp) then true; \
           else echo "Possible problem with ex2_1, diffs above"; fi; \
           ${RM} -f ex2_1.tmp
runex2_2:
	-@${MPIRUN} -np 1 ex2 -p 1 -snes_smonitor -snes_type umls -ksp_gmres_irorthog > ex2_2.tmp 2>&1;\
           if (diff output/ex2_2.out ex2_2.tmp) then true; \
           else echo "Possible problem with ex2_2, diffs above"; fi; \
           ${RM} -f ex2_2.tmp
runex2_3:
	-@${MPIRUN} -np 1 ex2 -p 1 -snes_smonitor -snes_type umls -snes_mf -ksp_gmres_irorthog > ex2_3.tmp 2>&1;\
           if (diff output/ex2_3.out ex2_3.tmp) then true; \
           else echo "Possible problem with ex2_3, diffs above"; fi; \
           ${RM} -f ex2_3.tmp
runex3:
	-@${MPIRUN} -np 2 ex3 -snes_smonitor > ex3_1.tmp 2>&1;\
           if (diff output/ex3_1.out ex3_1.tmp) then true; \
           else echo "Possible problem with ex3_1, diffs above"; fi;\
	   ${RM} -f ex3_1.tmp
runex3_2:
	-@${MPIRUN} -np 2 ex3 -Ny 2 -snes_type umls -snes_smonitor > ex3_2.tmp 2>&1;\
           if (diff output/ex3_2.out ex3_2.tmp) then true; \
           else echo "Possible problem with ex3_2, diffs above"; fi;\
	   ${RM} -f ex3_2.tmp
runex3_3:
	-@${MPIRUN} -np 4 ex3 -Nx 2 -Ny 2 -snes_fd -snes_smonitor > ex3_3.tmp 2>&1;\
           if (diff output/ex3_3.out ex3_3.tmp) then true; \
           else echo "Possible problem with ex3_3, diffs above"; fi;\
	   ${RM} -f ex3_3.tmp
