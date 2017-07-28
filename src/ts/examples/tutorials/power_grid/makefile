
CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/power_grid/
EXAMPLESC       = ex1.c ex2.c ex3.c ex3adj.c ex3opt.c ex3opt_fd.c ex5.c \
                ex6.c ex7.c ex8.c ex9adj.c ex9.c ex9opt.c
EXAMPLESF       =
EXAMPLESFH      =
MANSEC          = TS
DIRS            = stability_9bus
CLEANFILES      = SA-data/*

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_TS_LIB}
	${RM} ex1.o

ex1f: ex1f.o  chkopts
	-${FLINKER} -o ex1f ex1f.o  ${PETSC_TS_LIB}
	${RM} ex1f.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_TS_LIB}
	${RM} ex2.o

ex2f: ex2f.o  chkopts
	-${FLINKER} -o ex2f ex2f.o   ${PETSC_TS_LIB}
	${RM} ex2f.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_TS_LIB}
	${RM} ex3.o

ex3fwd: ex3fwd.o  chkopts
	-${CLINKER} -o ex3fwd ex3fwd.o  ${PETSC_TS_LIB}

ex3adj: ex3adj.o  chkopts
	-${CLINKER} -o ex3adj ex3adj.o  ${PETSC_TS_LIB}
	${RM} ex3adj.o

ex3opt: ex3opt.o  chkopts
	-${CLINKER} -o ex3opt ex3opt.o  ${PETSC_LIB}
	${RM} ex3opt.o

ex3opt_fd: ex3opt_fd.o  chkopts
	-${CLINKER} -o ex3opt_fd ex3opt_fd.o  ${PETSC_LIB}
	${RM} ex3opt_fd.o

ex3opt_fwd: ex3opt_fwd.o  chkopts
	-${CLINKER} -o ex3opt_fwd ex3opt_fwd.o  ${PETSC_LIB}

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_TS_LIB}
	${RM} ex4.o

ex5: ex5.o  chkopts
	-${CLINKER} -o ex5 ex5.o  ${PETSC_TS_LIB}
	${RM} ex5.o

ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_TS_LIB}
	${RM} ex6.o

ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_TS_LIB}
	${RM} ex7.o

ex8: ex8.o  chkopts
	-${CLINKER} -o ex8 ex8.o  ${PETSC_TS_LIB}
	${RM} ex8.o

ex9: ex9.o  chkopts
	-${CLINKER} -o ex9 ex9.o  ${PETSC_TS_LIB}
	${RM} ex9.o

ex9adj: ex9adj.o  chkopts
	-${CLINKER} -o ex9adj ex9adj.o  ${PETSC_TS_LIB}
	${RM} ex9adj.o

ex9opt: ex9opt.o  chkopts
	-${CLINKER} -o ex9opt ex9opt.o  ${PETSC_LIB}
	${RM} ex9opt.o

#---------------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 -ksp_gmres_cgs_refinement_type refine_always -snes_type newtonls -ts_max_steps 10  > ex1_1.tmp 2>&1;  \
	   ${DIFF} output/ex1_1.out ex1_1.tmp || printf "${PWD}\nPossible problem with ex1_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex1_1.tmp
runex1_2:
	-@${MPIEXEC} -n 1 ./ex1  -ts_max_steps 10  > ex1_2.tmp 2>&1;  \
	   ${DIFF} output/ex1_2.out ex1_2.tmp || printf "${PWD}\nPossible problem with ex1_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex1_2.tmp

runex1f:
	-@${MPIEXEC} -n 1 ./ex1f -ksp_gmres_cgs_refinement_type refine_always -snes_type newtonls -ts_monitor_pseudo > ex1f_1.tmp 2>&1;	  \
	   ${DIFF} output/ex1f_1.out ex1f_1.tmp || printf "${PWD}\nPossible problem with ex1f_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex1f_1.tmp

runex2:
	-@${MPIEXEC} -n 1 ./ex2 -nox -ts_dt 10  > ex2_1.tmp 2>&1;  \
	   ${DIFF} output/ex2_1.out ex2_1.tmp || printf "${PWD}\nPossible problem with ex2_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex2_1.tmp

runex2f:
	-@${MPIEXEC} -n 1 ./ex2f -ts_max_steps 100 > ex2f_1.tmp 2>&1; \
	  ${DIFF} output/ex2f_1.out ex2f_1.tmp || printf "${PWD}\nPossible problem with ex2f_1, diffs above\n=========================================\n"; \
	  ${RM} -f ex2_1.tmp

runex3:
	-@${MPIEXEC} -n 1 ./ex3 -nox  > ex3_1.tmp 2>&1;  \
	   ${DIFF} output/ex3_1.out ex3_1.tmp || printf "${PWD}\nPossible problem with ex3_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex3_1.tmp

runex3fwd:
	-@${MPIEXEC} -n 1 ./ex3fwd -ts_type cn -pc_type lu > ex3fwd_1.tmp 2>&1;    \
	   ${DIFF} output/ex3fwd_1.out ex3fwd_1.tmp || printf "${PWD}\nPossible problem with ex3fwd_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex3fwd_1.tmp

runex3adj:
	-@${MPIEXEC} -n 1 ./ex3adj -viewer_binary_skip_info -ts_type cn -pc_type lu > ex3adj_1.tmp 2>&1;  \
	   ${DIFF} output/ex3adj_1.out ex3adj_1.tmp || printf "${PWD}\nPossible problem with ex3adj_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex3adj_1.tmp SA-data/*

runex3opt:
	-@${MPIEXEC} -n 1 ./ex3opt -viewer_binary_skip_info -ts_type cn -pc_type lu -tao_monitor > ex3opt_1.tmp 2>&1;  \
	   ${DIFF} output/ex3opt_1.out ex3opt_1.tmp || printf "${PWD}\nPossible problem with ex3opt_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex3opt_1.tmp SA-data/*

runex3opt_fd:
	-@${MPIEXEC} -n 1 ./ex3opt_fd -ts_type cn -pc_type lu -tao_monitor -tao_gatol 1e-3 > ex3opt_fd_1.tmp 2>&1;  \
	   ${DIFF} output/ex3opt_fd_1.out ex3opt_fd_1.tmp || printf "${PWD}\nPossible problem with ex3opt_fd_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex3opt_fd_1.tmp

runex3opt_fwd:
	-@${MPIEXEC} -n 1 ./ex3opt_fwd -ts_type cn -pc_type lu -tao_monitor > ex3opt_fwd_1.tmp 2>&1;  \
	   ${DIFF} output/ex3opt_fwd_1.out ex3opt_fwd_1.tmp || printf "${PWD}\nPossible problem with ex3opt_fwd_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex3opt_fwd_1.tmp

runex4:
	-@${MPIEXEC} -n 1 ./ex4 -ts_view -nox > ex4_1.tmp 2>&1;  \
	   if (${DIFF} output/ex4_1.out ex4_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex4_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex4_1.tmp
runex4_2:
	-@${MPIEXEC} -n 3 ./ex4 -ts_view -nox > ex4_2.tmp 2>&1;  \
	   if (${DIFF} output/ex4_2.out ex4_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex4_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex4_2.tmp
runex4_3:
	-@${MPIEXEC} -n 1 ./ex4 -ts_view -nox -nonlinear > ex4_3.tmp 2>&1;  \
	   if (${DIFF} output/ex4_3.out ex4_3.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex4_3, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex4_3.tmp
runex4_4:
	-@${MPIEXEC} -n 3 ./ex4 -ts_view -nox -nonlinear > ex4_4.tmp 2>&1;  \
	   if (${DIFF} output/ex4_4.out ex4_4.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex4_4, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex4_4.tmp
runex4_sundials:
	-@${MPIEXEC} -n 4 ./ex4 -nox -ts_type sundials -ts_max_steps 500 -nonlinear > ex4.tmp 2>&1;  \
	   if (${DIFF} output/ex4_sundials.out ex4.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex4_sundials, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex4.tmp

runex6:
	-@${MPIEXEC} -n 1 ./ex6 -nox -time_steps_max 4   > ex6_1.tmp 2>&1;  \
	   if (${DIFF} output/ex6_1.out ex6_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex6_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex6_1.tmp

runex7:
	-@${MPIEXEC} -n 1 ./ex7 -max_steps 5   > ex7_1.tmp 2>&1;  \
	   if (${DIFF} output/ex7_1.out ex7_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex7_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex7_1.tmp

runex7_2:
	-@${MPIEXEC} -n 1 ./ex7 -max_steps 5  -snes_mf_operator > ex7_2.tmp 2>&1;  \
	   if (${DIFF} output/ex7_2.out ex7_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex7_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex7_2.tmp

runex7_3:
	-@${MPIEXEC} -n 1 ./ex7 -max_steps 5  -snes_mf > ex7_3.tmp 2>&1;  \
	   if (${DIFF} output/ex7_3.out ex7_3.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex7_3, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex7_3.tmp

runex9:
	-@${MPIEXEC} -n 1 ./ex9 > ex9_1.tmp 2>&1; \
	   ${DIFF} output/ex9_1.out ex9_1.tmp || printf "${PWD}\nPossible problem with ex9_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex9_1.tmp

runex9adj:
	-@${MPIEXEC} -n 1 ./ex9adj -viewer_binary_skip_info -ts_adapt_type none  > ex9adj_1.tmp 2>&1; \
	   ${DIFF} output/ex9adj_1.out ex9adj_1.tmp || printf "${PWD}\nPossible problem with ex9adj_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex9adj_1.tmp SA-data/*

runex9opt:
	-@${MPIEXEC} -n 1 ./ex9opt -viewer_binary_skip_info -ts_adapt_type none -tao_monitor -tao_gatol 0.0 -tao_grtol 1.e-3 -tao_converged_reason > ex9opt_1.tmp 2>&1; \
	 ${DIFF} output/ex9opt_1.out ex9opt_1.tmp || printf "${PWD}\nPossible problem with ex9opt_1, diffs above\n=========================================\n"; \
	 ${RM} -f ex9opt_1.tmp SA-data/*

runex9opt_2:
	-@${MPIEXEC} -n 1 ./ex9opt -viewer_binary_skip_info -Pm 1.1 -ts_adapt_type none -tao_type test -tao_test_gradient -tao_test_display -tao_monitor > ex9opt_2.tmp 2>&1;\
	 ${DIFF} output/ex9opt_2.out ex9opt_2.tmp || printf "${PWD}\nPossible problem with ex9opt_2, diffs above\n=========================================\n"; \
	 ${RM} -f ex9opt_2.tmp SA-data/*

TESTEXAMPLES_C_NOCOMPLEX          = ex2.PETSc runex2 ex2.rm  \
                                    ex5.PETSc ex5.rm \
                                    ex6.PETSc ex6.rm ex7.PETSc  ex7.rm \
                                    ex9adj.PETSc runex9adj ex9adj.rm \
                                    ex9opt.PETSc runex9opt ex9opt.rm
TESTEXAMPLES_C_NOCOMPLEX_NOTSINGLE = ex1.PETSc runex1 runex1_2 ex1.rm \
                                    ex3.PETSc runex3 ex3.rm ex3fwd.PETSc runex3fwd ex3fwd.rm ex3adj.PETSc runex3adj ex3adj.rm \
                                    ex3opt_fd.PETSc runex3opt_fd ex3opt_fd.rm ex3opt.PETSc runex3opt ex3opt.rm ex3opt_fwd.PETSc runex3opt_fwd ex3opt_fwd.rm
TESTEXAMPLES_C_X =
TESTEXAMPLES_FORTRAN =
TESTEXAMPLES_C_X_MPIUNI =
TESTEXAMPLES_13 =

include ${PETSC_DIR}/lib/petsc/conf/test
