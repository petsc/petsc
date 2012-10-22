
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c \
                ex9.c ex10.c ex12.c ex13.c ex14.c ex15.c ex16.c ex17.c \
                ex18.c ex19.c ex20.c ex21.c ex22.c ex23.c ex24.c ex25.c ex26.c
EXAMPLESF       = ex1f.F ex2f.F ex22f.F # ex22f_mf.F90
EXAMPLESFH      = ex2f.h
MANSEC          = TS
DIRS            = phasefield

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

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

ex10: ex10.o  chkopts
	-${CLINKER} -o ex10 ex10.o  ${PETSC_TS_LIB}
	${RM} ex10.o

ex11: ex11.o  chkopts
	-${CLINKER} -o ex11 ex11.o  ${PETSC_TS_LIB}
	${RM} ex11.o

ex12: ex12.o  chkopts
	-${CLINKER} -o ex12 ex12.o  ${PETSC_TS_LIB}
	${RM} ex12.o

ex13: ex13.o  chkopts
	-${CLINKER} -o ex13 ex13.o  ${PETSC_TS_LIB}
	${RM} ex13.o

ex14: ex14.o  chkopts
	-${CLINKER} -o ex14 ex14.o  ${PETSC_TS_LIB}
	${RM} ex14.o

ex15: ex15.o  chkopts
	-${CLINKER} -o ex15 ex15.o  ${PETSC_TS_LIB}
	${RM} ex15.o

ex16: ex16.o  chkopts
	-${CLINKER} -o ex16 ex16.o  ${PETSC_TS_LIB}
	${RM} ex16.o

ex17: ex17.o  chkopts
	-${CLINKER} -o ex17 ex17.o  ${PETSC_TS_LIB}
	${RM} ex17.o
ex18: ex18.o  chkopts
	-${CLINKER} -o ex18 ex18.o  ${PETSC_TS_LIB}
	${RM} ex18.o

ex19: ex19.o  chkopts
	-${CLINKER} -o ex19 ex19.o  ${PETSC_TS_LIB}
	${RM} ex19.o

ex19a: ex19a.o  chkopts
	-${CLINKER} -o ex19a ex19a.o  ${PETSC_TS_LIB}
	${RM} ex19a.o

ex20: ex20.o  chkopts
	-${CLINKER} -o ex20 ex20.o  ${PETSC_TS_LIB}
	${RM} ex20.o

ex21: ex21.o  chkopts
	-${CLINKER} -o ex21 ex21.o  ${PETSC_TS_LIB}
	${RM} ex21.o 

ex22: ex22.o chkopts
	-${CLINKER} -o ex22 ex22.o ${PETSC_TS_LIB}
	${RM} ex22.o

ex22f: ex22f.o  chkopts
	-${FLINKER} -o ex22f ex22f.o ${PETSC_TS_LIB}
	${RM} ex22f.o

ex22f_mf: ex22f_mf.o  chkopts
	-${FLINKER} -o ex22f_mf ex22f_mf.o ${PETSC_TS_LIB}
	${RM} ex22f_mf.o

ex23: ex23.o chkopts
	-${CLINKER} -o ex23 ex23.o ${PETSC_TS_LIB}
	${RM} ex23.o

ex24: ex24.o chkopts
	-${CLINKER} -o ex24 ex24.o ${PETSC_TS_LIB}
	${RM} ex24.o

ex25: ex25.o chkopts
	-${CLINKER} -o ex25 ex25.o ${PETSC_TS_LIB}
	${RM} ex25.o

ex26: ex26.o chkopts
	-${CLINKER} -o ex26 ex26.o ${PETSC_TS_LIB}
	${RM} ex26.o

ex26aspin: ex26aspin.o chkopts
	-${CLINKER} -o ex26aspin ex26aspin.o ${PETSC_TS_LIB}
	${RM} ex26aspin.o

ex27: ex27.o chkopts
	-${CLINKER} -o ex27 ex27.o ${PETSC_TS_LIB}
	${RM} ex27.o

#---------------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 -ksp_gmres_cgs_refinement_type refine_always -snes_type ls -ts_monitor_pseudo > ex1_1.tmp 2>&1;	  \
	   ${DIFF} output/ex1_1.out ex1_1.tmp || echo  ${PWD} "\nPossible problem with ex1_1, diffs above \n========================================="; \
	   ${RM} -f ex1_1.tmp
runex1_2:
	-@${MPIEXEC} -n 1 ./ex1 -ts_monitor_pseudo > ex1_2.tmp 2>&1;	  \
	   ${DIFF} output/ex1_2.out ex1_2.tmp || echo  ${PWD} "\nPossible problem with ex1_2, diffs above \n========================================="; \
	   ${RM} -f ex1_2.tmp

runex1f:
	-@${MPIEXEC} -n 1 ./ex1f -ksp_gmres_cgs_refinement_type refine_always -snes_type ls -ts_monitor_pseudo > ex1f_1.tmp 2>&1;	  \
	   ${DIFF} output/ex1f_1.out ex1f_1.tmp || echo  ${PWD} "\nPossible problem with ex1f_1, diffs above \n========================================="; \
	   ${RM} -f ex1f_1.tmp

runex2:
	-@${MPIEXEC} -n 2 ./ex2 -nox -ts_dt 10 -mymonitor > ex2_1.tmp 2>&1;	  \
	   ${DIFF} output/ex2_1.out ex2_1.tmp || echo  ${PWD} "\nPossible problem with ex2_1, diffs above \n========================================="; \
	   ${RM} -f ex2_1.tmp

runex2f:
	-@${MPIEXEC} -n 1 ./ex2f -ts_max_steps 100 > ex2f_1.tmp 2>&1; \
	  ${DIFF} output/ex2f_1.out ex2f_1.tmp || echo  ${PWD} "\nPossible problem with ex2_1, diffs above \n========================================="; \
	  ${RM} -f ex2_1.tmp

runex3:
	-@${MPIEXEC} -n 1 ./ex3 -nox -ts_type ssp -ts_dt 0.0005 > ex3_1.tmp 2>&1;	  \
	   ${DIFF} output/ex3_1.out ex3_1.tmp || echo  ${PWD} "\nPossible problem with ex3_1, diffs above \n========================================="; \
	   ${RM} -f ex3_1.tmp
runex3_2:
	-@${MPIEXEC} -n 1 ./ex3 -nox -ts_type ssp -ts_dt 0.0005 -time_dependent_rhs 1 > ex3_2.tmp 2>&1;	  \
	   ${DIFF} output/ex3_1.out ex3_2.tmp || echo  ${PWD} "\nPossible problem with ex3_2, diffs above \n========================================="; \
	   ${RM} -f ex3_2.tmp

runex4:
	-@${MPIEXEC} -n 1 ./ex4 -ts_view -nox > ex4_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex4_1.out ex4_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex4_1, diffs above \n========================================="; fi; \
	   ${RM} -f ex4_1.tmp
runex4_2:
	-@${MPIEXEC} -n 3 ./ex4 -ts_view -nox > ex4_2.tmp 2>&1;	  \
	   if (${DIFF} output/ex4_2.out ex4_2.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex4_2, diffs above \n========================================="; fi; \
	   ${RM} -f ex4_2.tmp
runex4_3:
	-@${MPIEXEC} -n 1 ./ex4 -ts_view -nox -nonlinear > ex4_3.tmp 2>&1;	  \
	   if (${DIFF} output/ex4_3.out ex4_3.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex4_3, diffs above \n========================================="; fi; \
	   ${RM} -f ex4_3.tmp
runex4_4:
	-@${MPIEXEC} -n 3 ./ex4 -ts_view -nox -nonlinear > ex4_4.tmp 2>&1;	  \
	   if (${DIFF} output/ex4_4.out ex4_4.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex4_4, diffs above \n========================================="; fi; \
	   ${RM} -f ex4_4.tmp
runex4_sundials:
	-@${MPIEXEC} -n 4 ./ex4 -nox -ts_type sundials -ts_max_steps 500 -nonlinear > ex4.tmp 2>&1;	  \
	   if (${DIFF} output/ex4_sundials.out ex4.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex4_sundials, diffs above \n========================================="; fi; \
	   ${RM} -f ex4.tmp

runex6:
	-@${MPIEXEC} -n 1 ./ex6 -nox -time_steps_max 4   > ex6_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex6_1.out ex6_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex6_1, diffs above \n========================================="; fi; \
	   ${RM} -f ex6_1.tmp

runex7:
	-@${MPIEXEC} -n 1 ./ex7 -max_steps 5   > ex7_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex7_1.out ex7_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex7_1, diffs above \n========================================="; fi; \
	   ${RM} -f ex7_1.tmp

runex7_2:
	-@${MPIEXEC} -n 1 ./ex7 -max_steps 5  -snes_mf_operator > ex7_2.tmp 2>&1;	  \
	   if (${DIFF} output/ex7_2.out ex7_2.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex7_2, diffs above \n========================================="; fi; \
	   ${RM} -f ex7_2.tmp

runex7_3:
	-@${MPIEXEC} -n 1 ./ex7 -max_steps 5  -snes_mf > ex7_3.tmp 2>&1;	  \
	   if (${DIFF} output/ex7_3.out ex7_3.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex7_3, diffs above \n========================================="; fi; \
	   ${RM} -f ex7_3.tmp

runex9:
	-@${MPIEXEC} -n 1 ./ex9 -da_grid_x 100 -initial 1 -xmin -2 -xmax 5 -exact -limit mc > ex9_1.tmp 2>&1; \
	   ${DIFF} output/ex9_1.out ex9_1.tmp || echo  ${PWD} "\nPossible problem with ex9_1, diffs above \n========================================="; \
	   ${RM} -f ex9_1.tmp
runex9_2:
	-@${MPIEXEC} -n 1 ./ex9 -da_grid_x 100 -initial 2 -xmin -2 -xmax 2 -exact -limit mc -physics burgers -bc_type outflow -ts_final_time 1 > ex9_2.tmp 2>&1; \
	   ${DIFF} output/ex9_2.out ex9_2.tmp || echo  ${PWD} "\nPossible problem with ex9_2, diffs above \n========================================="; \
	   ${RM} -f ex9_2.tmp
runex9_3:
	-@${MPIEXEC} -n 3 ./ex9 -da_grid_x 100 -initial 2 -xmin -2 -xmax 2 -exact -limit mc -physics burgers -bc_type outflow -ts_final_time 1 > ex9_3.tmp 2>&1; \
	   ${DIFF} output/ex9_3.out ex9_3.tmp || echo  ${PWD} "\nPossible problem with ex9_3, diffs above \n========================================="; \
	   ${RM} -f ex9_3.tmp

runex10:
	-@${MPIEXEC} -n 1 ./ex10 -da_grid_x 20 -rd_initial 1 -rd_discretization fd \
	   -rd_jacobian fd_coloring -rd_endpoint -ts_final_time 3 -ts_dt 1e-1 -ts_theta_extrapolate 0 \
	   -ts_monitor -snes_monitor_short -ksp_monitor_short > ex10_1.tmp 2>&1; \
	   ${DIFF} output/ex10_1.out ex10_1.tmp || echo  ${PWD} "\nPossible problem with ex10_1, diffs above \n========================================="; \
	   ${RM} -f ex10_1.tmp

runex10_2:
	-@${MPIEXEC} -n 1 ./ex10 -da_grid_x 20 -rd_initial 1 -rd_discretization fe -rd_quadrature lobatto2 \
	   -rd_jacobian fd_coloring -rd_endpoint -ts_final_time 3 -ts_dt 1e-1 -ts_theta_extrapolate 0 \
	   -ts_monitor -snes_monitor_short -ksp_monitor_short > ex10_2.tmp 2>&1; \
	   ${DIFF} output/ex10_2.out ex10_2.tmp || echo  ${PWD} "\nPossible problem with ex10_2, diffs above \n========================================="; \
	   ${RM} -f ex10_2.tmp

runex10_3:
	-@${MPIEXEC} -n 2 ./ex10 -da_grid_x 20 -rd_initial 1 -rd_discretization fd \
	   -rd_jacobian analytic -rd_endpoint -ts_final_time 3 -ts_dt 1e-1 -ts_theta_extrapolate 0 \
	   -ts_monitor -snes_monitor_short -ksp_monitor_short > ex10_3.tmp 2>&1; \
	   ${DIFF} output/ex10_3.out ex10_3.tmp || echo  ${PWD} "\nPossible problem with ex10_3, diffs above \n========================================="; \
	   ${RM} -f ex10_3.tmp

#  This currently crashes after a few time steps with nonsolvable nonlinear system
runex10_4:
	-@${MPIEXEC} -n 2 ./ex10 -da_grid_x 20 -rd_initial 1 -rd_discretization fe -rd_quadrature lobatto2 \
	   -rd_jacobian analytic -rd_endpoint -ts_final_time 3 -ts_dt 1e-1 -ts_theta_extrapolate 0 \
	   -ts_monitor -snes_monitor_short -ksp_monitor_short > ex10_4.tmp 2>&1; \
	   ${DIFF} output/ex10_4.out ex10_4.tmp || echo  ${PWD} "\nPossible problem with ex10_4, diffs above \n========================================="; \
	   ${RM} -f ex10_4.tmp

runex12:
	-@${MPIEXEC} -n 1 ./ex12 -da_grid_x 20 -ts_final_time 3 -ts_dt 1e-1 -ts_theta_extrapolate 0 \
	   -ts_monitor -ksp_monitor_short > ex12_1.tmp 2>&1; \
	   ${DIFF} output/ex12_1.out ex12_1.tmp || echo  ${PWD} "\nPossible problem with ex12_1, diffs above \n========================================="; \
	   ${RM} -f ex12_1.tmp

runex13:
	-@${MPIEXEC} -n 1 ./ex13 -ts_max_steps 5   > ex13.tmp 2>&1; \
	   ${DIFF} output/ex13_1.out ex13.tmp || echo  ${PWD} "\nPossible problem with ex13, diffs above \n========================================="; \
	   ${RM} -f ex13.tmp
runex13_2:
	-@${MPIEXEC} -n 3 ./ex13 -ts_max_steps 5   > ex13.tmp 2>&1; \
	   ${DIFF} output/ex13_1.out ex13.tmp || echo  ${PWD} "\nPossible problem with ex13_2, diffs above \n========================================="; \
	   ${RM} -f ex13.tmp
runex13_3:
	-@${MPIEXEC} -n 3 ./ex13 -ts_max_steps 5 -use_coloring  > ex13.tmp 2>&1; \
	   ${DIFF} output/ex13_1.out ex13.tmp || echo  ${PWD} "\nPossible problem with ex13_3, diffs above \n========================================="; \
	   ${RM} -f ex13.tmp

runex15:
	-@${MPIEXEC} -n 1 ./ex15 -da_grid_x 20 -da_grid_y 20 -boundary 0 -ts_max_steps 10 > ex15.tmp 2>&1; \
	   ${DIFF} output/ex15_1.out ex15.tmp || echo  ${PWD} "\nPossible problem with ex15_1, diffs above \n========================================="; \
	   ${RM} -f ex15.tmp
runex15_2:
	-@${MPIEXEC} -n 1 ./ex15 -da_grid_x 20 -da_grid_y 20 -boundary 0 -ts_max_steps 10 -Jtype 2 > ex15.tmp 2>&1; \
	   ${DIFF} output/ex15_1.out ex15.tmp || echo  ${PWD} "\nPossible problem with ex15_2, diffs above \n========================================="; \
	   ${RM} -f ex15.tmp
runex15_3:
	-@${MPIEXEC} -n 1 ./ex15 -da_grid_x 20 -da_grid_y 20 -boundary 1 -ts_max_steps 10 > ex15.tmp 2>&1; \
	   ${DIFF} output/ex15_3.out ex15.tmp || echo  ${PWD} "\nPossible problem with ex15_3, diffs above \n========================================="; \
	   ${RM} -f ex15.tmp
runex15_4:
	-@${MPIEXEC} -n 2 ./ex15 -da_grid_x 20 -da_grid_y 20 -boundary 1 -ts_max_steps 10 > ex15.tmp 2>&1; \
	   ${DIFF} output/ex15_4.out ex15.tmp || echo  ${PWD} "\nPossible problem with ex15_4, diffs above \n========================================="; \
	   ${RM} -f ex15.tmp
runex15_5:
	-@${MPIEXEC} -n 1 ./ex15 -da_grid_x 20 -da_grid_y 20 -boundary 0 -ts_max_steps 10 -Jtype 1 > ex15.tmp 2>&1; \
	   ${DIFF} output/ex15_1.out ex15.tmp || echo  ${PWD} "\nPossible problem with ex15_2, diffs above \n========================================="; \
	   ${RM} -f ex15.tmp

runex16:
	-@${MPIEXEC} -n 1 ./ex16 -ts_type arkimex -ts_arkimex_type myark2 > ex16_1.tmp 2>&1; \
	   ${DIFF} output/ex16_1.out ex16_1.tmp || echo  ${PWD} "\nPossible problem with ex16_1, diffs above \n========================================="; \
	   ${RM} -f ex16_1.tmp

runex17:
	-@${MPIEXEC} -n 1 ./ex17 -da_grid_x 40 -ts_max_steps 2 -snes_monitor_short -ksp_monitor_short > ex17_1.tmp 2>&1; \
	   ${DIFF} output/ex17_1.out ex17_1.tmp || echo  ${PWD} "\nPossible problem with ex17_1, diffs above \n========================================="; \
	   ${RM} -f ex17_1.tmp

runex17_2:
	-@${MPIEXEC} -n 1 ./ex17 -da_grid_x 100 -ts_type theta -ts_theta_theta 0.5 > ex17_2.tmp 2>&1; \
	   ${DIFF} output/ex17_2.out ex17_2.tmp || echo  ${PWD} "\nPossible problem with ex17_2, diffs above \n=========================================";  \
	   ${RM} -f ex17_2.tmp

runex22:
	-@${MPIEXEC} -n 1 ./ex22 -da_grid_x 200 -ts_monitor_solution -ts_arkimex_type 4 -ts_adapt_type none > ex22_1.tmp 2>&1; \
	   ${DIFF} output/ex22_1.out ex22_1.tmp || echo  ${PWD} "\nPossible problem with ex22_1, diffs above \n=========================================";  \
	   ${RM} -f ex22_1.tmp
runex22_2:
	-@${MPIEXEC} -n 2 ./ex22 -da_grid_x 200 -ts_monitor_solution -ts_type rosw -ts_dt 1e-3 -ts_adapt_type none > ex22_2.tmp 2>&1; \
	   ${DIFF} output/ex22_2.out ex22_2.tmp || echo  ${PWD} "\nPossible problem with ex22_2, diffs above \n=========================================";  \
	   ${RM} -f ex22_2.tmp
runex22_3:
	-@${MPIEXEC} -n 2 ./ex22 -da_grid_x 200 -ts_monitor_solution -ts_type rosw -ts_rosw_type ra34pw2 -ts_dt 5e-3 -ts_adapt_type none > ex22_3.tmp 2>&1; \
	   ${DIFF} output/ex22_3.out ex22_3.tmp || echo  ${PWD} "\nPossible problem with ex22_3, diffs above \n=========================================";  \
	   ${RM} -f ex22_3.tmp
runex22f:
	-@${MPIEXEC} -n 1 ./ex22f -da_grid_x 200 -ts_monitor_solution -ts_arkimex_type 4 > ex22f_1.tmp 2>&1; \
	   ${DIFF} output/ex22f_1.out ex22f_1.tmp || echo  ${PWD} "\nPossible problem with ex22f_1, diffs above \n=========================================";  \
	   ${RM} -f ex22f_1.tmp

runex22f_mf:
	-@${MPIEXEC} -n 1 ./ex22f_mf -da_grid_x 200 -ts_monitor_solution -ts_arkimex_type 4 > ex22f_mf_1.tmp 2>&1; \
	   ${DIFF} output/ex22f_mf_1.out ex22f_mf_1.tmp || echo  ${PWD} "\nPossible problem with ex22f_mf_1, diffs above \n=========================================";  \
	   ${RM} -f ex22f_mf_1.tmp

runex24:                        # This is not in the nightlies because the convergence details are numerically sensitive
	-@${MPIEXEC} -n 1 ./ex24 -pc_type lu -ts_dt 1e-5 -ts_final_time 1e5 -n 50 -monitor_short > ex24_1.tmp 2>&1; \
	   ${DIFF} output/ex24_1.out ex24_1.tmp || echo  ${PWD} "\nPossible problem with ex24_1, diffs above \n=========================================";  \
	   ${RM} -f ex24_1.tmp

runex25:
	-@${MPIEXEC} -n 1 ./ex25 -da_grid_x 20 -ts_monitor_solution -ts_type rosw -ts_rosw_type 2p -ts_dt 5e-2 -ts_adapt_type none > ex25_1.tmp 2>&1; \
	   ${DIFF} output/ex25_1.out ex25_1.tmp || echo  ${PWD} "\nPossible problem with ex25_1, diffs above \n=========================================";  \
	   ${RM} -f ex25_1.tmp

runex26:
	-@${MPIEXEC} -n 1 ./ex26 -da_grid_x 20 -da_grid_y 20 -lidvelocity 100 -grashof 1e3 -ts_max_steps 100 -ts_rtol 1e-3 -ts_atol 1e-3 -ts_type rosw -ts_rosw_type ra3pw -ts_monitor -ts_monitor_solution_vtk 'foo-%03D.vts' > ex26_1.tmp 2>&1; \
	   ${DIFF} output/ex26_1.out ex26_1.tmp || echo  ${PWD} "\nPossible problem with ex26_1, diffs above \n=========================================";  \
	   ${RM} -f ex26_1.tmp

runex26aspin:
	-@${MPIEXEC} -n 4 ./ex26aspin -da_refine 3 -ts_max_steps 1 -ts_rtol 1e-3 -ts_atol 1e-3 -ts_type beuler -ts_monitor -snes_monitor -snes_mf -ksp_monitor -overlap 2


TESTEXAMPLES_C		  = ex1.PETSc runex1 runex1_2 ex1.rm ex2.PETSc runex2 ex2.rm ex3.PETSc runex3 runex3_2 ex3.rm \
                            ex4.PETSc runex4 runex4_2 runex4_3 runex4_4 ex4.rm ex5.PETSc ex5.rm \
                            ex6.PETSc runex6 ex6.rm ex7.PETSc runex7 runex7_2 runex7_3 ex7.rm \
                            ex10.PETSc runex10 runex10_2 runex10_3  ex10.rm \
                            ex12.PETSc ex12.rm ex13.PETSc runex13 runex13_2 runex13_3 ex13.rm\
                            ex15.PETSc runex15 runex15_2 runex15_3 runex15_4 runex15_5 ex15.rm \
                            ex17.PETSc runex17 runex17_2 ex17.rm ex22.PETSc ex22.rm
TESTEXAMPLES_C_X	  = 
TESTEXAMPLES_FORTRAN	  = ex1f.PETSc runex1f ex1f.rm ex2f.PETSc runex2f ex2f.rm ex22f.PETSc ex22f.rm # ex22f_mf.PETSc ex22f_mf.rm
TESTEXAMPLES_C_X_MPIUNI = 
TESTEXAMPLES_13		  = ex2.PETSc ex2.rm ex3.PETSc ex3.rm ex4.PETSc ex4.rm \
                            ex5.PETSc ex5.rm

include ${PETSC_DIR}/conf/test
