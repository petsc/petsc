
CFLAGS	        = -g
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/power_grid/stability_9bus/
EXAMPLESC       = ex9bus.c ex9busadj.c ex9busopt.c ex9busopt_fd.c ex9busdmnetwork.c
EXAMPLESF       =
EXAMPLESFH      =
MANSEC          = TS
DIRS            =
CLEANFILES      = SA-data/*

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

ex9busdmnetwork: ex9busdmnetwork.o  chkopts
	-${CLINKER} -o ex9busdmnetwork ex9busdmnetwork.o  ${PETSC_TS_LIB}

#---------------------------------------------------------------------------------
runex9bus_implicit:
	-@${MPIEXEC} -n 1 ./ex9bus -ts_monitor -snes_monitor_short > ex9bus_1.tmp 2>&1;    \
	   ${DIFF} output/ex9bus_1.out ex9bus_1.tmp || printf "${PWD}\nPossible problem with ex9bus_implicit, diffs above\n=========================================\n"; \
	   ${RM} -f ex9bus_1.tmp SA-data/* out.bin

runex9bus_semiexplicit:
	-@x="bad"; ${MPIEXEC} -n 1 ./ex9bus -ts_monitor -snes_monitor_short -dae_semiexplicit -ts_rk_type 2a > ex9bus_2.tmp 2>&1;   \
           if (${DIFF} output/ex9bus_2.out ex9bus_2.tmp > /dev/null 2>&1) then x='good'; fi ;\
           if (${DIFF} output/ex9bus_2_alt.out ex9bus_2.tmp > /dev/null 2>&1) then x='good'; fi; \
           if (${DIFF} output/ex9bus_2_alt2.out ex9bus_2.tmp > /dev/null 2>&1) then x='good'; fi; \
           if [ "$$x" = "bad" ]; then ${DIFF} output/ex9bus_2.out ex9bus_2.tmp ; ${DIFF} output/ex9bus_2_alt.out ex9bus_2.tmp ; printf "${PWD}\nPossible problem with ex9bus_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex9bus_2.tmp SA-data/* out.bin

runex9bus_steprestart:
	-@${MPIEXEC} -n 1 ./ex9bus -ts_monitor -snes_monitor_short -ts_type arkimex > ex9bus_3.tmp 2>&1;    \
	   ${DIFF} output/ex9bus_3.out ex9bus_3.tmp || printf "${PWD}\nPossible problem with ex9bus_steprestart, diffs above\n=========================================\n"; \
	   ${RM} -f ex9bus_3.tmp SA-data/* out.bin

runex9busadj:
	-@${MPIEXEC} -n 1 ./ex9busadj -viewer_binary_skip_info > ex9busadj_1.tmp 2>&1;	  \
	   ${DIFF} output/ex9busadj_1.out ex9busadj_1.tmp || printf "${PWD}\nPossible problem with ex9busadj_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex9busadj_1.tmp SA-data/*

runex9busopt:
	-@${MPIEXEC} -n 1 ./ex9busopt -viewer_binary_skip_info -tao_monitor -tao_gttol .2 > ex9busopt_1.tmp 2>&1;	  \
	   ${DIFF} output/ex9busopt_1.out ex9busopt_1.tmp || printf "${PWD}\nPossible problem with ex9busopt_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex9busopt_1.tmp SA-data/*

runex9busopt_fd:
	-@${MPIEXEC} -n 1 ./ex9busopt_fd -viewer_binary_skip_info -tao_monitor -tao_gttol .2 > ex9busopt_fd_1.tmp 2>&1;	  \
	   ${DIFF} output/ex9busopt_fd_1.out ex9busopt_fd_1.tmp || printf "${PWD}\nPossible problem with ex9busopt_fd_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex9busopt_fd_1.tmp SA-data/*

runex9busdmnetwork:
	-@${MPIEXEC} -n 1 ./ex9busdmnetwork -ts_monitor -snes_converged_reason -alg_snes_converged_reason > ex9bus.tmp 2>&1;	  \
	   ${DIFF} output/ex9busdmnetwork_1.out ex9bus.tmp || printf "${PWD}\nPossible problem with ex9busdmnetwork, diffs above\n=========================================\n"; \
	   ${RM} -f ex9bus.tmp SA-data/*

runex9busdmnetwork_2:
	-@${MPIEXEC} -n 2 ./ex9busdmnetwork -ts_monitor -snes_converged_reason -alg_snes_converged_reason > ex9bus.tmp 2>&1;	  \
	   ${DIFF} output/ex9busdmnetwork_2.out ex9bus.tmp || printf "${PWD}\nPossible problem with ex9busdmnetwork_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex9bus.tmp SA-data/*

TESTEXAMPLES_DOUBLEINT32 = ex9bus.PETSc runex9bus_implicit runex9bus_semiexplicit runex9bus_steprestart ex9bus.rm \
                           ex9busadj.PETSc runex9busadj ex9busadj.rm ex9busopt.PETSc runex9busopt ex9busopt.rm \
                           ex9busopt_fd.PETSc runex9busopt_fd ex9busopt_fd.rm

include ${PETSC_DIR}/lib/petsc/conf/test
