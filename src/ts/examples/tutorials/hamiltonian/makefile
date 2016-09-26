
CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/hamiltonian
EXAMPLESC       = ex1.c
EXAMPLESF       =
EXAMPLESFH      =
MANSEC          = TS

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_TS_LIB}
	${RM} ex1.o

runex1:
	-@${MPIEXEC} -n 1 ./ex1 -monitor > ex1_1.tmp 2>&1; \
	  ${DIFF} output/ex1_1.out ex1_1.tmp || printf "${PWD}\nPossible problem with ex1_1, diffs above\n=========================================\n"; \
	  ${RM} -f ex1_1.tmp

runex1_2:
	-@${MPIEXEC} -n 1 ./ex1 -monitor -ts_bsi_type velverlet > ex1_2.tmp 2>&1; \
	  ${DIFF} output/ex1_2.out ex1_2.tmp || printf "${PWD}\nPossible problem with ex1_2, diffs above\n=========================================\n"; \
	  ${RM} -f ex1_2.tmp

runex1_3:
	-@${MPIEXEC} -n 1 ./ex1 -monitor -ts_type euler

runex1_4:
	-@${MPIEXEC} -n 1 ./ex1 -monitor -ts_type rk

TESTEXAMPLES_C		  = ex1.PETSc runex1 ex1.rm
TESTEXAMPLES_C_X	  =

include ${PETSC_DIR}/lib/petsc/conf/test
