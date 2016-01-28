
CFLAGS	        = -g
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/power_grid/stability_9bus/
EXAMPLESC       = ex9bus.c ex9busadj.c ex9busopt.c ex9busopt_fd.c
EXAMPLESF       =
EXAMPLESFH      =
MANSEC          = TS
DIRS            =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex9bus: ex9bus.o  chkopts
	-${CLINKER} -o ex9bus ex9bus.o  ${PETSC_TS_LIB}
	${RM} ex9bus.o

ex9busadj: ex9busadj.o  chkopts
	-${CLINKER} -o ex9busadj ex9busadj.o  ${PETSC_TS_LIB}
	${RM} ex9busadj.o

ex9busopt: ex9busopt.o  chkopts
	-${CLINKER} -o ex9busopt ex9busopt.o  ${PETSC_LIB}

ex9busopt_fd: ex9busopt_fd.o  chkopts
	-${CLINKER} -o ex9busopt_fd ex9busopt_fd.o  ${PETSC_LIB}

runex9bus:
	-@${MPIEXEC} -n 1 ./ex9bus

runex9busadj:
	-@${MPIEXEC} -n 1 ./ex9busadj -viewer_binary_skip_info > ex9busadj_1.tmp 2>&1;	  \
	   ${DIFF} output/ex9busadj_1.out ex9busadj_1.tmp || printf "${PWD}\nPossible problem with ex9busadj_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex9busadj_1.tmp

runex9busopt:
	-@${MPIEXEC} -n 1 ./ex9busopt -viewer_binary_skip_info -tao_monitor -tao_gttol .2 > ex9busopt_1.tmp 2>&1;	  \
	   ${DIFF} output/ex9busopt_1.out ex9busopt_1.tmp || printf "${PWD}\nPossible problem with ex9busopt_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex9busopt_1.tmp

runex9busopt_fd:
	-@${MPIEXEC} -n 1 ./ex9busopt_fd -viewer_binary_skip_info -tao_monitor -tao_gttol .2 > ex9busopt_fd_1.tmp 2>&1;	  \
	   ${DIFF} output/ex9busopt_fd_1.out ex9busopt_fd_1.tmp || printf "${PWD}\nPossible problem with ex9busopt_fd_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex9busopt_fd_1.tmp

clean_files:
	${RM} ex9busopt.o
	-@${RM} ex9bus-SA-*

TESTEXAMPLES_DOUBLEINT32 = ex9bus.PETSc runex9bus ex9bus.rm ex9busadj.PETSc runex9busadj ex9busadj.rm ex9busopt.PETSc runex9busopt ex9busopt.rm ex9busopt_fd.PETSc runex9busopt_fd ex9busopt_fd.rm

include ${PETSC_DIR}/lib/petsc/conf/test
