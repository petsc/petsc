
CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/hybrid/
EXAMPLESC       = ex1.c
EXAMPLESF       =
EXAMPLESFH      =
MANSEC          = TS
DIRS            =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_TS_LIB}
	${RM} ex1.o

ex1adj: ex1adj.o  chkopts
	-${CLINKER} -o ex1adj ex1adj.o  ${PETSC_TS_LIB}
	${RM} ex1adj.o

ex1fd: ex1fd.o  chkopts
	-${CLINKER} -o ex1fd ex1fd.o  ${PETSC_TS_LIB}
	${RM} ex1fd.o

ex1fwd: ex1fwd.o  chkopts
	-${CLINKER} -o ex1fwd ex1fwd.o  ${PETSC_TS_LIB}
	${RM} ex1fwd.o

#---------------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 -ts_monitor > ex1_1.tmp 2>&1;    \
    ${DIFF} output/ex1_1.out ex1_1.tmp || printf "${PWD}\nPossible problem with ex1_1, diffs above\n=========================================\n"; \
    ${RM} -f ex1_1.tmp

runex1_2:
	-@${MPIEXEC} -n 1 ./ex1 -ts_monitor_lg_solution -1 -draw_pause -2

runex1adj:
	-@${MPIEXEC} -n 1 ./ex1adj -ts_monitor -ts_adjoint_monitor > ex1adj_1.tmp 2>&1;    \
	${DIFF} output/ex1adj_1.out ex1adj_1.tmp || printf "${PWD}\nPossible problem with ex1adj_1, diffs above\n=========================================\n"; \
	${RM} -f ex1adj_1.tmp

runex1adj_2:
	-@${MPIEXEC} -n 1 ./ex1adj -ts_monitor_lg_solution -1 -draw_pause -2 -ts_monitor -ts_adjoint_monitor -ts_event_tol 1e-9

runex1fd:
	-@${MPIEXEC} -n 1 ./ex1fd -ts_event_tol 1e-9

runex1fwd:
	-@${MPIEXEC} -n 1 ./ex1fwd -ts_monitor > ex1fwd_1.tmp 2>&1;    \
	${DIFF} output/ex1fwd_1.out ex1fwd_1.tmp || printf "${PWD}\nPossible problem with ex1fwd_1, diffs above\n=========================================\n"; \
	${RM} -f ex1fwd_1.tmp

runex1fwd_2:
	-@${MPIEXEC} -n 1 ./ex1fwd -ts_monitor_lg_solution -1 -draw_pause -2 -ts_monitor -ts_event_tol 1e-9

clean_files:
	-@${RM} *.bin; \
	  ${RM} *.info

TESTEXAMPLES_C_NOCOMPLEX = ex1.PETSc runex1 ex1.rm \
                           ex1adj.PETSc runex1adj ex1adj.rm \
                           ex1fwd.PETSc runex1fwd ex1fwd.rm

TESTEXAMPLES_C_X  =
TESTEXAMPLES_FORTRAN  =
TESTEXAMPLES_C_X_MPIUNI =
TESTEXAMPLES_13 =

include ${PETSC_DIR}/lib/petsc/conf/test
