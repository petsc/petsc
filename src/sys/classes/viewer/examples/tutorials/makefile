
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR  = src/sys/classes/viewer/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c ex3.c
EXAMPLESF       = ex1f90.F90
MANSEC          = Viewer

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_SYS_LIB}
	${RM} ex1.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_SYS_LIB}
	${RM} ex2.o

ex1f90: ex1f90.o  chkopts
	-${FLINKER} -o ex1f90 ex1f90.o ${PETSC_SYS_LIB}
	${RM} ex1f90.o

ex3: ex3.o chkopts
	-${CLINKER} -o ex3 ex3.o ${PETSC_LIB}
	${RM} ex3.o

#----------------------------------------------------------------------------

runex1f90:
	-@${MPIEXEC} -n 1 ./ex1f90 > ex1f90_1.tmp 5>&1; \
	   if (${DIFF} output/ex1f90_1.out ex1f90_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1f90_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1f90_1.tmp

runex3_1:
	-@${MPIEXEC} -n 1 ./ex3 > ex3.tmp 2>&1; \
	   if (${DIFF} output/ex3_1_p1.out ex3.tmp) then true; \
	   else printf "${PWD}\nPossible problem with with ex3_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex3.tmp

runex3_2:
	-@${MPIEXEC} -n 6 ./ex3 > ex3.tmp 2>&1; \
	   if (${DIFF} output/ex3_1_p6.out ex3.tmp) then true; \
	   else printf "${PWD}\nPossible problem with with ex3_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex3.tmp

runex3_3:
	-@${MPIEXEC} -n 12 ./ex3 > ex3.tmp 2>&1; \
	   if (${DIFF} output/ex3_1_p12.out ex3.tmp) then true; \
	   else printf "${PWD}\nPossible problem with with ex3_3, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex3.tmp

runex3_mpiio:
	-@${MPIEXEC} -n 6 ./ex3 -usempiio > ex3.tmp 2>&1; \
	   if (${DIFF} output/ex3_2_p6.out ex3.tmp) then true; \
	   else printf "${PWD}\nPossible problem with with ex3_mpiio, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex3.tmp


TESTEXAMPLES_C          = ex2.PETSc ex2.rm \
			  ex3.PETSc runex3_1 runex3_2 runex3_3 runex3_mpiio ex3.rm
TESTEXAMPLES_F90        =
TESTEXAMPLES_C_X_MPIUNI =

include ${PETSC_DIR}/conf/test

