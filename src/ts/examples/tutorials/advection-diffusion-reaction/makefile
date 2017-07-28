
CLFAGS          =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/advection-diffusion-reaction/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex7.c ex10.c
EXAMPLESF       = shashi.F90
EXAMPLESFH      =
MANSEC          = TS
DIRS            =
CLEANFILES      =  SA-data/*

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

shashi: shashi.o  chkopts
	-${FLINKER} -o shashi shashi.o  ${PETSC_TS_LIB}
	${RM} shashi.o

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_TS_LIB}
	${RM} ex1.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_TS_LIB}
	${RM} ex2.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_TS_LIB}
	${RM} ex3.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_TS_LIB}
	${RM} ex4.o

ex5: ex5.o  chkopts
	-${CLINKER} -o ex5 ex5.o  ${PETSC_TS_LIB}
	${RM} ex5.o

ex5adj: ex5adj.o  chkopts
	-${CLINKER} -o ex5adj ex5adj.o  ${PETSC_TS_LIB}
	${RM} ex5adj.o

ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_TS_LIB}
	${RM} ex6.o

ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_TS_LIB}
	${RM} ex7.o

ex10: ex10.o  chkopts
	-${CLINKER} -o ex10 ex10.o  ${PETSC_TS_LIB}
	${RM} ex10.o

#---------------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 -ksp_gmres_cgs_refinement_type refine_always -snes_type newtonls -ts_monitor_pseudo > ex1_1.tmp 2>&1;	  \
	   ${DIFF} output/ex1_1.out ex1_1.tmp || printf "${PWD}\nPossible problem with ex1_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex1_1.tmp
runex1_2:
	-@${MPIEXEC} -n 1 ./ex1 -ts_monitor_pseudo > ex1_2.tmp 2>&1;	  \
	   ${DIFF} output/ex1_2.out ex1_2.tmp || printf "${PWD}\nPossible problem with ex1_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex1_2.tmp

runex2:
	-@${MPIEXEC} -n 2 ./ex2 -nox -ts_dt 10 -mymonitor > ex2_1.tmp 2>&1;	  \
	   ${DIFF} output/ex2_1.out ex2_1.tmp || printf "${PWD}\nPossible problem with ex2_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex2_1.tmp

runex3:
	-@${MPIEXEC} -n 1 ./ex3 -nox -ts_type ssp -ts_dt 0.0005 > ex3_1.tmp 2>&1;	  \
	   ${DIFF} output/ex3_1.out ex3_1.tmp || printf "${PWD}\nPossible problem with ex3_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex3_1.tmp
runex4:
	-@${MPIEXEC} -n 1 ./ex4 -ts_view -nox > ex4_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex4_1.out ex4_1.tmp) then true; \
	   else printf "${PWD}\n\Possible problem with ex4_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex4_1.tmp
runex4_2:
	-@${MPIEXEC} -n 3 ./ex4 -ts_view -nox > ex4_2.tmp 2>&1;	  \
	   if (${DIFF} output/ex4_2.out ex4_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex4_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex4_2.tmp
runex4_3:
	-@${MPIEXEC} -n 1 ./ex4 -ts_view -nox -nonlinear > ex4_3.tmp 2>&1;	  \
	   if (${DIFF} output/ex4_3.out ex4_3.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex4_3, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex4_3.tmp
runex4_4:
	-@${MPIEXEC} -n 3 ./ex4 -ts_view -nox -nonlinear > ex4_4.tmp 2>&1;	  \
	   if (${DIFF} output/ex4_4.out ex4_4.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex4_4, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex4_4.tmp
runex4_sundials:
	-@${MPIEXEC} -n 4 ./ex4 -nox -ts_type sundials -ts_max_steps 500 -nonlinear > ex4.tmp 2>&1;	  \
	   if (${DIFF} output/ex4_sundials.out ex4.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex4_sundials, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex4.tmp

runex5:
	-@${MPIEXEC} -n 1 ./ex5 -ts_max_steps 10 -ts_monitor   > ex5_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex5_1.out ex5_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex5_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex5_1.tmp

runex5adj:
	-@${MPIEXEC} -n 1 ./ex5adj -ts_max_steps 10 -ts_monitor -ts_adjoint_monitor -da_grid_x 16 -da_grid_y 16 > ex5adj_1.tmp 2>&1; \
	   if (${DIFF} output/ex5adj_1.out ex5adj_1.tmp) then true; \
	   else printf "${PWD}\n Possible problem with ex5adj_1, diffs above\n=======================================\n"; fi; \
	   ${RM} -f ex5adj_1.tmp SA-data/*

runex5adj_2:
	-@${MPIEXEC} -n 2 ./ex5adj -ts_max_steps 10 -ts_dt 10 -ts_adjoint_monitor_draw_sensi -draw_pause -2

runex5adj_knl:
	-@${MPIEXEC} -n 1 ./ex5adj -ts_max_steps 10 -ts_monitor -ts_adjoint_monitor -da_grid_x 128 -da_grid_y 128 -ts_trajectory_type memory -ts_trajectory_solution_only 0 -malloc_hbw -ts_trajectory_use_dram 1 2>&1

runex6:
	-@${MPIEXEC} -n 1 ./ex6 -ts_max_steps 10 -ts_monitor > ex6_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex6_1.out ex6_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex6_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex6_1.tmp

runex6_2:
	-@${MPIEXEC} -n 3 ./ex6 -ts_max_steps 10 -ts_monitor > ex6_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex6_1.out ex6_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex6_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex6_1.tmp

runex6_3:
	-@${MPIEXEC} -n 1 ./ex6 -ts_max_steps 10 -ts_monitor -useLaxWendroff false > ex6_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex6_3.out ex6_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex6_3, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex6_1.tmp

runex6_4:
	-@${MPIEXEC} -n 3 ./ex6 -ts_max_steps 10 -ts_monitor -useLaxWendroff false > ex6_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex6_3.out ex6_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex6_4, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex6_1.tmp

runex7:
	-@${MPIEXEC} -n 1 ./ex7 -ts_max_steps 5 -ts_monitor -useLaxWendroff false > ex7_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex7_1.out ex7_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex7_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex7_1.tmp

runex7_2:
	-@${MPIEXEC} -n 1 ./ex7 -ts_max_steps 5 -ts_monitor -snes_mf_operator > ex7_2.tmp 2>&1;	  \
	   if (${DIFF} output/ex7_1.out ex7_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex7_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex7_2.tmp

runex7_3:
	-@${MPIEXEC} -n 1 ./ex7 -ts_max_steps 5 -ts_monitor  -snes_mf > ex7_3.tmp 2>&1;	  \
	   if (${DIFF} output/ex7_1.out ex7_3.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex7_3, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex7_3.tmp

runex10:
	-@${MPIEXEC} -n 1 ./ex10 -da_grid_x 20 -rd_initial 1 -rd_discretization fd \
	   -rd_jacobian fd_coloring -rd_endpoint -ts_final_time 3 -ts_dt 1e-1 -ts_theta_initial_guess_extrapolate 0 \
	   -ts_monitor -snes_monitor_short -ksp_monitor_short > ex10_1.tmp 2>&1; \
	   ${DIFF} output/ex10_1.out ex10_1.tmp || printf "${PWD}\nPossible problem with ex10_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex10_1.tmp

runex10_2:
	-@${MPIEXEC} -n 1 ./ex10 -da_grid_x 20 -rd_initial 1 -rd_discretization fe -rd_quadrature lobatto2 \
	   -rd_jacobian fd_coloring -rd_endpoint -ts_final_time 3 -ts_dt 1e-1 -ts_theta_initial_guess_extrapolate 0 \
	   -ts_monitor -snes_monitor_short -ksp_monitor_short > ex10_2.tmp 2>&1; \
	   ${DIFF} output/ex10_2.out ex10_2.tmp || printf "${PWD}\nPossible problem with ex10_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex10_2.tmp

runex10_3:
	-@${MPIEXEC} -n 2 ./ex10 -da_grid_x 20 -rd_initial 1 -rd_discretization fd \
	   -rd_jacobian analytic -rd_endpoint -ts_final_time 3 -ts_dt 1e-1 -ts_theta_initial_guess_extrapolate 0 \
	   -ts_monitor -snes_monitor_short -ksp_monitor_short > ex10_3.tmp 2>&1; \
	   ${DIFF} output/ex10_3.out ex10_3.tmp || printf "${PWD}\nPossible problem with ex10_3, diffs above\n=========================================\n"; \
	   ${RM} -f ex10_3.tmp

#  This currently crashes after a few time steps with nonsolvable nonlinear system
runex10_4:
	-@${MPIEXEC} -n 2 ./ex10 -da_grid_x 20 -rd_initial 1 -rd_discretization fe -rd_quadrature lobatto2 \
	   -rd_jacobian analytic -rd_endpoint -ts_final_time 3 -ts_dt 1e-1 -ts_theta_initial_guess_extrapolate 0 \
	   -ts_monitor -snes_monitor_short -ksp_monitor_short > ex10_4.tmp 2>&1; \
	   ${DIFF} output/ex10_4.out ex10_4.tmp || printf "${PWD}\nPossible problem with ex10_4, diffs above\n=========================================\n"; \
	   ${RM} -f ex10_4.tmp


TESTEXAMPLES_C		  = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex3.PETSc runex3 ex3.rm \
                            ex4.PETSc ex4.rm ex5.PETSc runex5 ex5.rm \
                            ex5adj.PETSc runex5adj ex5adj.rm ex6.PETSc runex6 runex6_2 runex6_3 runex6_4 ex6.rm
TESTEXAMPLES_C_X	  =
TESTEXAMPLES_FORTRAN	  =
TESTEXAMPLES_C_X_MPIUNI   =
TESTEXAMPLES_13		  =

include ${PETSC_DIR}/lib/petsc/conf/test
