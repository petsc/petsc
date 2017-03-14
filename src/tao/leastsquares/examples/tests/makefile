ALL:

CFLAGS		 =
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/tao/leastsquares/examples/tests/
MANSEC		 =
EXAMPLESC        = chwirut1.c chwirut2.c
DOCS		 =
DIRS		 =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

chwirut1: chwirut1.o chkopts
	-${CLINKER} -o chwirut1 chwirut1.o ${PETSC_TAO_LIB}
	${RM} chwirut1.o

chwirut2: chwirut2.o chkopts
	-${CLINKER} -o chwirut2 chwirut2.o ${PETSC_TAO_LIB}
	${RM} chwirut2.o

runchwirut1:
	-@${MPIEXEC} -n 1 ./chwirut1 -tao_smonitor -tao_max_it 100 -tao_type pounders > chwirut1_1.tmp 2>&1;\
	${DIFF} output/chwirut1_1.out chwirut1_1.tmp || printf '${PWD}\nPossible problem with chwirut1_1 stdout, diffs above \n=====================================\n';\
	${RM} -f chwirut1_1.tmp

runchwirut1_2:
	-@${MPIEXEC} -n 1 ./chwirut1 -tao_smonitor -tao_max_it 100 -wtype 1 -tao_type pounders > chwirut1_2.tmp 2>&1;\
        ${DIFF} output/chwirut1_2.out chwirut1_2.tmp || printf '${PWD}\nPossible problem with chwirut1_2 stdout, diffs above \n=====================================\n';\
	${RM} -f chwirut1_2.tmp

runchwirut1_3:
	-@${MPIEXEC} -n 1 ./chwirut1 -tao_smonitor -tao_max_it 100 -wtype 2 -tao_type pounders > chwirut1_3.tmp 2>&1;\
        ${DIFF} output/chwirut1_3.out chwirut1_3.tmp || printf '${PWD}\nPossible problem with chwirut1_3 stdout, diffs above \n=====================================\n';\
	${RM} -f chwirut1_3.tmp

runchwirut1_4:
	-@${MPIEXEC} -n 1 ./chwirut1 -tao_smonitor -tao_max_it 100 -tao_type pounders -pounders_subsolver_tao_type blmvm > chwirut1_4.tmp 2>&1;\
	${DIFF} output/chwirut1_4.out chwirut1_4.tmp || printf '${PWD}\nPossible problem with chwirut1_4 stdout, diffs above \n=====================================\n';\
	${RM} -f chwirut1_4.tmp


TESTEXAMPLES_C             = chwirut1.PETSc runchwirut1 chwirut1.rm
# 			     chwirut2.PETSc runchwirut2 chwirut2.rm
TESTEXAMPLES_C_X_MPIUNI    = chwirut1.PETSc runchwirut1 chwirut1.rm


include ${PETSC_DIR}/lib/petsc/conf/test


