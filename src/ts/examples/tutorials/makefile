
CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c \
                ex9.c ex10.c ex11.c ex12.c ex13.c ex14.c ex15.c ex16.c ex17.c \
                ex19.c ex20.c ex21.c ex22.c ex24.c ex25.c ex26.c \
                ex28.c ex31.c ex34.c ex35.cxx extchem.c\
                ex16adj.c ex16opt_p.c ex16opt_ic.c  \
                ex20adj.c ex20opt_p.c ex20opt_ic.c  \
                ex40.c ex41.c ex42.c
EXAMPLESF       = ex1f.F ex22f.F ex22f_mf.F90
MANSEC          = TS
DIRS            = phasefield advection-diffusion-reaction eimex power_grid network
CLEANFILES      = SA-data/*
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_TS_LIB}
	${RM} ex1.o

ex1f: ex1f.o  chkopts
	-${FLINKER} -o ex1f ex1f.o  ${PETSC_TS_LIB}
	${RM} ex1f.o

ex2: ex2.o chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_TS_LIB}
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

ex11_sa: ex11_sa.o chkopts
	-${CLINKER} -o ex11_sa ex11_sa.o ${PETSC_TS_LIB}
	${RM} ex11_sa.o

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

ex16adj: ex16adj.o chkopts
	-${CLINKER} -o ex16adj ex16adj.o ${PETSC_TS_LIB}
	${RM} ex16adj.o

ex16opt_p: ex16opt_p.o chkopts
	-${CLINKER} -o ex16opt_p ex16opt_p.o ${PETSC_LIB}
	${RM} ex16opt_p.o

ex16opt_ic: ex16opt_ic.o chkopts
	-${CLINKER} -o ex16opt_ic ex16opt_ic.o ${PETSC_LIB}
	${RM} ex16opt_ic.o

ex17: ex17.o  chkopts
	-${CLINKER} -o ex17 ex17.o  ${PETSC_TS_LIB}
	${RM} ex17.o

ex18: ex18.o  chkopts
	-${CLINKER} -o ex18 ex18.o  ${PETSC_TS_LIB}
	${RM} ex18.o

ex19: ex19.o  chkopts
	-${CLINKER} -o ex19 ex19.o  ${PETSC_TS_LIB}
	${RM} ex19.o

ex20: ex20.o  chkopts
	-${CLINKER} -o ex20 ex20.o  ${PETSC_TS_LIB}
	${RM} ex20.o

ex20adj: ex20adj.o  chkopts
	-${CLINKER} -o ex20adj ex20adj.o  ${PETSC_TS_LIB}
	${RM} ex20adj.o

ex20fwd: ex20fwd.o  chkopts
	-${CLINKER} -o ex20fwd ex20fwd.o  ${PETSC_TS_LIB}
	${RM} ex20fwd.o

ex20opt_ic: ex20opt_ic.o chkopts
	-${CLINKER} -o ex20opt_ic ex20opt_ic.o ${PETSC_LIB}
	${RM} ex20opt_ic.o

ex20opt_p: ex20opt_p.o chkopts
	-${CLINKER} -o ex20opt_p ex20opt_p.o ${PETSC_LIB}
	${RM} ex20opt_p.o

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

ex27: ex27.o chkopts
	-${CLINKER} -o ex27 ex27.o ${PETSC_TS_LIB}
	${RM} ex27.o

ex28: ex28.o chkopts
	-${CLINKER} -o ex28 ex28.o ${PETSC_TS_LIB}
	${RM} ex28.o

ex29: ex29.o chkopts
	-${CLINKER} -o ex29 ex29.o ${PETSC_TS_LIB}
	${RM} ex29.o

ex31: ex31.o chkopts
	-${CLINKER} -o ex31 ex31.o ${PETSC_TS_LIB}
	${RM} ex31.o

ex34: ex34.o chkopts
	-${CLINKER} -o ex34 ex34.o ${PETSC_TS_LIB}
	${RM} ex34.o

ex35: ex35.o chkopts
	-${CLINKER} -o ex35 ex35.o ${PETSC_TS_LIB}
	${RM} ex35.o

ex36: ex36.o chkopts
	-${CLINKER} -o ex36 ex36.o ${PETSC_TS_LIB}
	${RM} ex36.o

ex36SE: ex36SE.o chkopts
	-${CLINKER} -o ex36SE ex36SE.o ${PETSC_TS_LIB}
	${RM} ex36SE.o

ex36A: ex36A.o chkopts
	-${CLINKER} -o ex36A ex36A.o ${PETSC_TS_LIB}
	${RM} ex36A.o

extchem: extchem.o chkopts
	-${CLINKER} -o extchem extchem.o ${PETSC_TS_LIB}
	${RM} extchem.o

extchemfield: extchemfield.o chkopts
	-${CLINKER} -o extchemfield extchemfield.o ${PETSC_TS_LIB}
	${RM} extchemfield.o

ex40: ex40.o chkopts
	-${CLINKER} -o ex40 ex40.o ${PETSC_TS_LIB}
	${RM} ex40.o

ex41: ex41.o chkopts
	-${CLINKER} -o ex41 ex41.o ${PETSC_TS_LIB}
	${RM} ex41.o

ex42: ex42.o chkopts
	-${CLINKER} -o ex42 ex42.o ${PETSC_TS_LIB}
	${RM} ex42.o

ex43: ex43.o chkopts
	-${CLINKER} -o ex43 ex43.o ${PETSC_TS_LIB}
	${RM} ex43.o

ex44: ex44.o chkopts
	-${CLINKER} -o ex44 ex44.o ${PETSC_TS_LIB}
	${RM} ex44.o

#---------------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 -ksp_gmres_cgs_refinement_type refine_always -snes_type newtonls -ts_monitor_pseudo -snes_atol 1.e-7 -ts_pseudo_frtol 1.e-5 -ts_view draw:tikz:output/fig.tex > ex1_1.tmp 2>&1;	  \
	   ${DIFF} output/ex1_1.out ex1_1.tmp || printf "${PWD}\nPossible problem with ex1_1, diffs above\n=========================================\n"; \
	   ${DIFF} output/fig.tex output/fig.tex.out || printf "${PWD}\nPossible problem with ex1_1, diffs above\n=========================================\n"; \
         ${RM} -f ex1_1.tmp output/fig.tex

runex1_2:
	-@${MPIEXEC} -n 1 ./ex1 -ts_monitor_pseudo -ts_pseudo_frtol 1.e-5 > ex1_2.tmp 2>&1;  \
	   ${DIFF} output/ex1_2.out ex1_2.tmp || printf "${PWD}\nPossible problem with ex1_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex1_2.tmp

runex1f:
	-@${MPIEXEC} -n 1 ./ex1f -ksp_gmres_cgs_refinement_type refine_always -snes_type newtonls -ts_monitor_pseudo -ts_max_snes_failures 3 -ts_pseudo_frtol 1.e-5  -snes_stol 1e-5 > ex1f_1.tmp 2>&1;  \
	   ${DIFF} output/ex1f_1.out ex1f_1.tmp || printf "${PWD}\nPossible problem with ex1f_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex1f_1.tmp

runex2:
	-@${MPIEXEC} -n 2 ./ex2 -nox -ts_dt 10 -mymonitor > ex2_1.tmp 2>&1;  \
	   ${DIFF} output/ex2_1.out ex2_1.tmp || printf "${PWD}\nPossible problem with ex2_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex2_1.tmp

runex3:
	-@${MPIEXEC} -n 1 ./ex3 -nox -ts_type ssp -ts_dt 0.0005 > ex3_1.tmp 2>&1;  \
	   ${DIFF} output/ex3_1.out ex3_1.tmp || printf "${PWD}\nPossible problem with ex3_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex3_1.tmp
runex3_2:
	-@${MPIEXEC} -n 1 ./ex3 -nox -ts_type ssp -ts_dt 0.0005 -time_dependent_rhs 1 > ex3_2.tmp 2>&1;  \
	   ${DIFF} output/ex3_1.out ex3_2.tmp || printf "${PWD}\nPossible problem with ex3_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex3_2.tmp
runex3_3:                       # IMEX IJacobian for constant RHS Jacobian
	-@${MPIEXEC} -n 1 ./ex3 -nox -ts_type rosw -ts_max_steps 3 -ksp_converged_reason | ${SED} "s/ATOL/RTOL/g" > ex3_3.tmp 2>&1;	  \
	   ${DIFF} output/ex3_3.out ex3_3.tmp || printf "${PWD}\nPossible problem with ex3_3, diffs above\n=========================================\n"; \
	   ${RM} -f ex3_3.tmp
runex3_4:                       # non-IMEX IJacobian with constant RHS Jacobian
	-@${MPIEXEC} -n 1 ./ex3 -nox -ts_type beuler -ts_max_steps 3 -ksp_converged_reason | ${SED} "s/ATOL/RTOL/g" > ex3_4.tmp 2>&1;  \
	   ${DIFF} output/ex3_4.out ex3_4.tmp || printf "${PWD}\nPossible problem with ex3_4, diffs above\n=========================================\n"; \
	   ${RM} -f ex3_4.tmp
runex3_5:
	-@${MPIEXEC} -n 1 ./ex3 -nox -ts_type beuler -ts_max_steps 3 -ksp_converged_reason -time_dependent_rhs | ${SED} "s/ATOL/RTOL/g" > ex3_5.tmp 2>&1;  \
	   ${DIFF} output/ex3_5.out ex3_5.tmp || printf "${PWD}\nPossible problem with ex3_5, diffs above\n=========================================\n"; \
	   ${RM} -f ex3_5.tmp

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
runex5:
	-@${MPIEXEC} -n 1 ./ex5    > ex5_1.tmp 2>&1;  \
	   if (${DIFF} output/ex5_1.out ex5_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex5, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex5_1.tmp
runex5_nox:
	-@${MPIEXEC} -n 1 ./ex5 -nox   > ex5_1.tmp 2>&1;  \
	   if (${DIFF} output/ex5_1.out ex5_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex5_nox, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex5_1.tmp
runex6:
	-@${MPIEXEC} -n 1 ./ex6 -nox -ts_max_steps 4   > ex6_1.tmp 2>&1;  \
	   if (${DIFF} output/ex6_1.out ex6_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex6_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex6_1.tmp

runex7:
	-@${MPIEXEC} -n 1 ./ex7 -ts_max_steps 5   > ex7_1.tmp 2>&1;  \
	   if (${DIFF} output/ex7_1.out ex7_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex7_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex7_1.tmp

runex7_2:
	-@${MPIEXEC} -n 1 ./ex7 -ts_max_steps 5  -snes_mf_operator > ex7_2.tmp 2>&1;  \
	   if (${DIFF} output/ex7_2.out ex7_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex7_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex7_2.tmp

runex7_3:
	-@${MPIEXEC} -n 1 ./ex7 -ts_max_steps 5  -snes_mf > ex7_3.tmp 2>&1;  \
	   if (${DIFF} output/ex7_3.out ex7_3.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex7_3, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex7_3.tmp

runex8:
	-@${MPIEXEC} -n 1 ./ex8 -monitor_error -ts_atol 1e-2 -ts_rtol 1e-2 -ts_exact_final_time interpolate -ts_type arkimex > ex8_1.tmp 2>&1; \
	   ${DIFF} output/ex8_1.out ex8_1.tmp || printf "${PWD}\nPossible problem with ex8_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex8_1.tmp

runex8_2:
	-@${MPIEXEC} -n 1 ./ex8 -ts_atol 1e-2 -ts_rtol 1e-2 -ts_final_time 15 -ts_type arkimex -ts_arkimex_type 2e -problem_type orego -ts_arkimex_initial_guess_extrapolate 0 > ex8_2.tmp 2>&1; \
	   ${DIFF} output/ex8_2.out ex8_2.tmp || printf "${PWD}\nPossible problem with ex8_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex8_2.tmp

runex8_3:
	-@${MPIEXEC} -n 1 ./ex8 -ts_atol 1e-2 -ts_rtol 1e-2 -ts_final_time 15 -ts_type arkimex -ts_arkimex_type 2e -problem_type orego -ts_arkimex_initial_guess_extrapolate 1 > ex8_3.tmp 2>&1; \
	   ${DIFF} output/ex8_3.out ex8_3.tmp || printf "${PWD}\nPossible problem with ex8_3, diffs above\n=========================================\n"; \
	   ${RM} -f ex8_3.tmp

runex9:
	-@${MPIEXEC} -n 1 ./ex9 -da_grid_x 100 -initial 1 -xmin -2 -xmax 5 -exact -limit mc > ex9_1.tmp 2>&1; \
	   ${DIFF} output/ex9_1.out ex9_1.tmp || printf "${PWD}\nPossible problem with ex9_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex9_1.tmp
runex9_2:
	-@${MPIEXEC} -n 1 ./ex9 -da_grid_x 100 -initial 2 -xmin -2 -xmax 2 -exact -limit mc -physics burgers -bc_type outflow -ts_final_time 1 | sed "s/at 48/at 0/g"> ex9_2.tmp 2>&1; \
	   ${DIFF} output/ex9_2.out ex9_2.tmp || printf "${PWD}\nPossible problem with ex9_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex9_2.tmp
runex9_3:
	-@${MPIEXEC} -n 3 ./ex9 -da_grid_x 100 -initial 2 -xmin -2 -xmax 2 -exact -limit mc -physics burgers -bc_type outflow -ts_final_time 1 | sed "s/at 48/at 0/g" > ex9_3.tmp 2>&1; \
	   ${DIFF} output/ex9_3.out ex9_3.tmp || printf "${PWD}\nPossible problem with ex9_3, diffs above\n=========================================\n"; \
	   ${RM} -f ex9_3.tmp

runex10:
	-@${MPIEXEC} -n 1 ./ex10 -da_grid_x 20 -rd_initial 1 -rd_discretization fd \
	   -rd_jacobian fd_coloring -rd_endpoint -ts_final_time 1 -ts_dt 2e-1 -ts_theta_initial_guess_extrapolate 0 \
	   -ts_monitor -snes_monitor_short -ksp_monitor_short > ex10_1.tmp 2>&1; \
	   ${DIFF} output/ex10_1.out ex10_1.tmp || printf "${PWD}\nPossible problem with ex10_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex10_1.tmp

runex10_2:
	-@${MPIEXEC} -n 1 ./ex10 -da_grid_x 20 -rd_initial 1 -rd_discretization fe -rd_quadrature lobatto2 \
	   -rd_jacobian fd_coloring -rd_endpoint -ts_final_time 1 -ts_dt 2e-1 -ts_theta_initial_guess_extrapolate 0 \
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

runex11:
	-@${MPIEXEC} -n 1 ./ex11 -ufv_vtk_interval 0 -f ${PETSC_DIR}/share/petsc/datafiles/meshes/sevenside-quad-15.exo > ex11_1.tmp 2>&1; \
	   ${DIFF} output/ex11_1.out ex11_1.tmp || printf "${PWD}\nPossible problem with ex11_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex11_1.tmp

runex11_euler:
	-@${MPIEXEC} -n 1 ./ex11 -ufv_vtk_interval 1 -monitor density,energy -f -grid_size 2,1 -grid_bounds -1,1.,0.,1 -bc_wall 1,2,3,4 \
          -dm_refine_hierarchy 5 -physics euler -eu_type iv_shock -ufv_cfl 10 -eu_alpha 60. -eu_gamma 1.4 -eu_amach 2.02 -eu_rho2 3. -ts_final_time 1 \
          -ts_ssp_type rks2 -ts_ssp_nstages 10 -petscfv_type leastsquares -petsclimiter_type minmod
#           > ex11_1.tmp 2>&1; \
#	   ${DIFF} output/ex11_1.out ex11_euler.tmp || printf "${PWD}\nPossible problem with ex11_1, diffs above\n=========================================\n";
	   ${RM} -f ex11_1.tmp
#[-dm_plex_convert_type p4est]

runex11_advect:
	-@${MPIEXEC} -n 4 ./ex11 \
		-physics advect -advect_sol_type bump_cavity -advect_bump_center 0.,0.5 -advect_bump_radius 0.25 -advect_bump_type cos -ufv_cfl 10 \
		-petscfv_type leastsquares -petscfv_compute_gradients 1 -petsclimiter_type minmod \
		-monitor solution,error -ufv_vtk_interval 10 \
		-f -grid_bounds -1.,1.,-1.,1. -bc_outflow 1,2,3,4 -grid_size 64,64 \
		-ts_ssp_type rks2 -ts_ssp_nstages 10 -ts_final_time 1.


runex11_advect_p4est:
	-@${MPIEXEC} -n 4 ./ex11 \
		-physics advect -advect_sol_type bump_cavity -advect_bump_center 0.,0.5 -advect_bump_radius 0.25 -advect_bump_type cos -ufv_cfl 10 \
		-petscfv_type leastsquares -petscfv_compute_gradients 1 -petsclimiter_type minmod \
		-monitor solution,error \
		-f -grid_bounds -1.,1.,-1.,1. -bc_outflow 1,2,3,4 \
			-dm_type p4est -dm_forest_partition_overlap 1 -dm_forest_maximum_refinement 6 -dm_forest_minimum_refinement 2 -dm_forest_initial_refinement 4 \
		-ts_ssp_type rks2 -ts_ssp_nstages 10 -ts_final_time 1. \
		-ufv_use_amr -refine_vec_tagger_box 1.e-2,inf -coarsen_vec_tagger_box 0.,1.e-4 -refine_tag_view -coarsen_tag_view

runex11_advect_p4est_fine:
	-@${MPIEXEC} -n 4 ./ex11 \
		-physics advect -advect_sol_type bump_cavity -advect_bump_center 0.,0.5 -advect_bump_radius 0.25 -advect_bump_type cos -ufv_cfl 10 \
		-petscfv_type leastsquares -petscfv_compute_gradients 1 -petsclimiter_type minmod \
		-monitor solution,error \
		-f -grid_bounds -1.,1.,-1.,1. -bc_outflow 1,2,3,4 \
			-dm_type p4est -dm_forest_partition_overlap 1 -dm_forest_maximum_refinement 7 -dm_forest_minimum_refinement 3 -dm_forest_initial_refinement 4 \
		-ts_ssp_type rks2 -ts_ssp_nstages 10 -ts_final_time 1. \
		-ufv_use_amr -refine_vec_tagger_box 2.5e-3,inf -coarsen_vec_tagger_box 0,2.5e-5 -refine_tag_view -coarsen_tag_view

runex11_euler_p4est:
	-@${MPIEXEC} -n 4 ./ex11 -ufv_vtk_interval 1 -monitor density,energy -f -grid_size 2,1 -grid_bounds -1,1.,0.,1 -bc_wall 1,2,3,4 \
          -dm_type p4est -dm_forest_partition_overlap 1 -dm_forest_maximum_refinement 6 -dm_forest_minimum_refinement 2 -dm_forest_initial_refinement 2 -physics euler -eu_type iv_shock -ufv_cfl 10 -eu_alpha 60. -grid_skew_60 -eu_gamma 1.4 -eu_amach 2.02 -eu_rho2 3. -ts_final_time 1 \
          -ts_ssp_type rks2 -ts_ssp_nstages 10 -petscfv_type leastsquares -petscfv_compute_gradients 0 -petsclimiter_type minmod \
					-ufv_use_amr -ufv_view_initial_refinement -refine_vec_tagger_box 0.5,inf -coarsen_vec_tagger_box 0,1.e-2 -refine_tag_view -coarsen_tag_view -ts_final_time 0.1

runex12:
	-@${MPIEXEC} -n 1 ./ex12 -da_grid_x 20 -ts_final_time 3 -ts_dt 1e-1 -ts_theta_initial_guess_extrapolate 0 \
	   -ts_monitor -ksp_monitor_short > ex12_1.tmp 2>&1; \
	   ${DIFF} output/ex12_1.out ex12_1.tmp || printf "${PWD}\nPossible problem with ex12_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex12_1.tmp

runex12_2:
	-@${MPIEXEC} -n 1 ./ex12 -da_grid_x 20 -ts_final_time 0.11 -ts_dt 1e-1 -ts_type glle \
	   -ts_monitor -ksp_monitor_short > ex12_2.tmp 2>&1; \
	   ${DIFF} output/ex12_2.out ex12_2.tmp || printf "${PWD}\nPossible problem with ex12_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex12_2.tmp

runex13:
	-@${MPIEXEC} -n 1 ./ex13 -ts_max_steps 5 -ts_monitor > ex13.tmp 2>&1; \
	   ${DIFF} output/ex13_1.out ex13.tmp || printf "${PWD}\nPossible problem with ex13, diffs above\n=========================================\n"; \
	   ${RM} -f ex13.tmp
runex13_2:
	-@${MPIEXEC} -n 3 ./ex13 -ts_max_steps 5 -ts_monitor > ex13.tmp 2>&1; \
	   ${DIFF} output/ex13_1.out ex13.tmp || printf "${PWD}\nPossible problem with ex13_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex13.tmp
runex13_3:
	-@${MPIEXEC} -n 3 ./ex13 -ts_max_steps 5 -snes_fd_color -ts_monitor > ex13.tmp 2>&1; \
	   ${DIFF} output/ex13_1.out ex13.tmp || printf "${PWD}\nPossible problem with ex13_3, diffs above\n=========================================\n"; \
	   ${RM} -f ex13.tmp

runex15:
	-@${MPIEXEC} -n 1 ./ex15 -da_grid_x 20 -da_grid_y 20 -boundary 0 -ts_max_steps 10 -ts_monitor > ex15.tmp 2>&1; \
	   ${DIFF} output/ex15_1.out ex15.tmp || printf "${PWD}\nPossible problem with ex15_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex15.tmp
runex15_2:
	-@${MPIEXEC} -n 1 ./ex15 -da_grid_x 20 -da_grid_y 20 -boundary 0 -ts_max_steps 10 -Jtype 2 -ts_monitor > ex15.tmp 2>&1; \
	   ${DIFF} output/ex15_1.out ex15.tmp || printf "${PWD}\nPossible problem with ex15_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex15.tmp
runex15_3:
	-@${MPIEXEC} -n 1 ./ex15 -da_grid_x 20 -da_grid_y 20 -boundary 1 -ts_max_steps 10 -ts_monitor > ex15.tmp 2>&1; \
	   ${DIFF} output/ex15_1.out ex15.tmp || printf "${PWD}\nPossible problem with ex15_3, diffs above\n=========================================\n"; \
	   ${RM} -f ex15.tmp
runex15_4:
	-@${MPIEXEC} -n 2 ./ex15 -da_grid_x 20 -da_grid_y 20 -boundary 1 -ts_max_steps 10 -ts_monitor > ex15.tmp 2>&1; \
	   ${DIFF} output/ex15_1.out ex15.tmp || printf "${PWD}\nPossible problem with ex15_4, diffs above\n=========================================\n"; \
	   ${RM} -f ex15.tmp
runex15_5:
	-@${MPIEXEC} -n 1 ./ex15 -da_grid_x 20 -da_grid_y 20 -boundary 0 -ts_max_steps 10 -Jtype 1 -ts_monitor > ex15.tmp 2>&1; \
	   ${DIFF} output/ex15_1.out ex15.tmp || printf "${PWD}\nPossible problem with ex15_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex15.tmp

runex16:
	-@${MPIEXEC} -n 1 ./ex16 -ts_type arkimex -ts_arkimex_type myark2 -ts_adapt_type none > ex16_1.tmp 2>&1; \
	   ${DIFF} output/ex16_1.out ex16_1.tmp || printf "${PWD}\nPossible problem with ex16_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex16_1.tmp

runex16adj:
	-@${MPIEXEC} -n 1 ./ex16adj -monitor 0 -viewer_binary_skip_info  > ex16adj_1.tmp 2>&1; \
	   ${DIFF} output/ex16adj_1.out ex16adj_1.tmp || printf "${PWD}\nPossible problem with ex16adj_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex16adj_1.tmp SA-data/*

runex16adj_2:
	-@${MPIEXEC} -n 1 ./ex16adj -monitor 0 -ts_trajectory_type memory > ex16adj_2.tmp 2>&1; \
	  ${DIFF} output/ex16adj_1.out ex16adj_2.tmp || printf "${PWD}\nPossible problem with ex16adj_2, diffs above\n=========================================\n"; \
	  ${RM} -f ex16adj_2.tmp SA-data/*

runex16opt_p:
	-@${MPIEXEC} -n 1 ./ex16opt_p -monitor 0 -viewer_binary_skip_info -tao_view -tao_monitor  -tao_gttol 1.e-5 > ex16opt_p_1.tmp 2>&1; \
	   ${DIFF} output/ex16opt_p_1.out ex16opt_p_1.tmp || printf "${PWD}\nPossible problem with ex16opt_p_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex16opt_p_1.tmp SA-data/*


runex16opt_ic:
	-@${MPIEXEC} -n 1 ./ex16opt_ic -monitor 0 -viewer_binary_skip_info -tao_view -tao_monitor  -tao_gttol 1.e-5 > ex16opt_ic_1.tmp 2>&1; \
	   ${DIFF} output/ex16opt_ic_1.out ex16opt_ic_1.tmp || printf "${PWD}\nPossible problem with ex16opt_ic_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex16opt_ic_1.tmp SA-data/*

runex17:
	-@${MPIEXEC} -n 1 ./ex17 -da_grid_x 40 -ts_max_steps 2 -snes_monitor_short -ksp_monitor_short -ts_monitor > ex17_1.tmp 2>&1; \
	   ${DIFF} output/ex17_1.out ex17_1.tmp || printf "${PWD}\nPossible problem with ex17_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex17_1.tmp

runex17_2:
	-@${MPIEXEC} -n 1 ./ex17 -da_grid_x 100 -ts_type theta -ts_theta_theta 0.5 > ex17_2.tmp 2>&1; \
	   ${DIFF} output/ex17_2.out ex17_2.tmp || printf "${PWD}\nPossible problem with ex17_2, diffs above\n=========================================\n";  \
	   ${RM} -f ex17_2.tmp

#  ex20 examples are not in the repository
runex20:
	-@${MPIEXEC} -n 1 ./ex20 > ex20_1.tmp 2>&1; \
	   ${DIFF} output/ex20_1.out ex20_1.tmp || printf "${PWD}\nPossible problem with ex20_1, diffs above\n=========================================\n";  \
	   ${RM} -f ex20_1.tmp

runex20adj:
	-@${MPIEXEC} -n 1 ./ex20adj -monitor 0 -ts_type theta -ts_theta_endpoint -ts_theta_theta 0.5 -viewer_binary_skip_info -ts_dt 0.001 -mu 100000 > ex20adj_1.tmp 2>&1; \
	  ${DIFF} output/ex20adj_1.out ex20adj_1.tmp || printf "${PWD}\nPossible problem with ex20adj_1, diffs above\n=========================================\n"; \
	  ${RM} -f ex20adj_1.tmp SA-data/*

runex20adj_1:
	-@${MPIEXEC} -n 1 ./ex20adj -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_solution_only > ex20adj.tmp 2>&1; \
	  ${DIFF} output/ex20adj_2.out ex20adj.tmp || printf "${PWD}\nPossible problem with ex20adj_1, diffs above\n=========================================\n"; \
	  ${RM} -f ex20adj.tmp SA-data/*

runex20adj_2:
	-@${MPIEXEC} -n 1 ./ex20adj -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_solution_only 0 > ex20adj.tmp 2>&1; \
	  ${DIFF} output/ex20adj_2.out ex20adj.tmp || printf "${PWD}\nPossible problem with ex20adj_2, diffs above\n=========================================\n"; \
	  ${RM} -f ex20adj.tmp SA-data/*

runex20adj_3:
	-@${MPIEXEC} -n 1 ./ex20adj -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_stride 5 -ts_trajectory_solution_only -ts_trajectory_save_stack > ex20adj.tmp 2>&1; \
	  ${DIFF} output/ex20adj_2.out ex20adj.tmp || printf "${PWD}\nPossible problem with ex20adj_3, diffs above\n=========================================\n"; \
	  ${RM} -f ex20adj.tmp SA-data/*

runex20adj_4:
	-@${MPIEXEC} -n 1 ./ex20adj -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_stride 5 -ts_trajectory_solution_only 0 -ts_trajectory_save_stack > ex20adj.tmp 2>&1; \
	  ${DIFF} output/ex20adj_2.out ex20adj.tmp || printf "${PWD}\nPossible problem with ex20adj_4, diffs above\n=========================================\n"; \
	  ${RM} -f ex20adj.tmp SA-data/*

runex20adj_5:
	-@${MPIEXEC} -n 1 ./ex20adj -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_stride 5 -ts_trajectory_solution_only -ts_trajectory_save_stack 0 > ex20adj.tmp 2>&1; \
	  ${DIFF} output/ex20adj_2.out ex20adj.tmp || printf "${PWD}\nPossible problem with ex20adj_5, diffs above\n=========================================\n"; \
	  ${RM} -f ex20adj.tmp SA-data/*

runex20adj_6:
	-@${MPIEXEC} -n 1 ./ex20adj -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_stride 5 -ts_trajectory_solution_only 0 -ts_trajectory_save_stack 0 > ex20adj.tmp 2>&1; \
	  ${DIFF} output/ex20adj_2.out ex20adj.tmp || printf "${PWD}\nPossible problem with ex20adj_6, diffs above\n=========================================\n"; \
	  ${RM} -f ex20adj.tmp SA-data/*

runex20adj_7:
	-@${MPIEXEC} -n 1 ./ex20adj -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 5 -ts_trajectory_solution_only -ts_trajectory_monitor > ex20adj.tmp 2>&1; \
	  ${DIFF} output/ex20adj_3.out ex20adj.tmp || printf "${PWD}\nPossible problem with ex20adj_7, diffs above\n=========================================\n"; \
	  ${RM} -f ex20adj.tmp SA-data/*

runex20adj_8:
	-@${MPIEXEC} -n 1 ./ex20adj -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 5 -ts_trajectory_solution_only 0 -ts_trajectory_monitor > ex20adj.tmp 2>&1; \
	  ${DIFF} output/ex20adj_4.out ex20adj.tmp || printf "${PWD}\nPossible problem with ex20adj_8, diffs above\n=========================================\n"; \
	  ${RM} -f ex20adj.tmp SA-data/*

runex20adj_9:
	-@${MPIEXEC} -n 1 ./ex20adj -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 5 -ts_trajectory_revolve_online -ts_trajectory_solution_only > ex20adj.tmp 2>&1; \
	  ${DIFF} output/ex20adj_2.out ex20adj.tmp || printf "${PWD}\nPossible problem with ex20adj_9, diffs above\n=========================================\n"; \
	  ${RM} -f ex20adj.tmp SA-data/*

runex20adj_10:
	-@${MPIEXEC} -n 1 ./ex20adj -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 5 -ts_trajectory_revolve_online -ts_trajectory_solution_only 0 > ex20adj.tmp 2>&1; \
	  ${DIFF} output/ex20adj_2.out ex20adj.tmp || printf "${PWD}\nPossible problem with ex20adj_10, diffs above\n=========================================\n"; \
	  ${RM} -f ex20adj.tmp SA-data/*

runex20adj_11:
	-@${MPIEXEC} -n 1 ./ex20adj -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_max_cps_disk 8 -ts_trajectory_solution_only > ex20adj.tmp 2>&1; \
	  ${DIFF} output/ex20adj_2.out ex20adj.tmp || printf "${PWD}\nPossible problem with ex20adj_11, diffs above\n=========================================\n"; \
	  ${RM} -f ex20adj.tmp SA-data/*

runex20adj_12:
	-@${MPIEXEC} -n 1 ./ex20adj -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_max_cps_disk 8 -ts_trajectory_solution_only 0 > ex20adj.tmp 2>&1; \
	  ${DIFF} output/ex20adj_2.out ex20adj.tmp || printf "${PWD}\nPossible problem with ex20adj_12, diffs above\n=========================================\n"; \
	  ${RM} -f ex20adj.tmp SA-data/*

runex20adj_13:
	-@${MPIEXEC} -n 1 ./ex20adj -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_stride 5 -ts_trajectory_solution_only -ts_trajectory_save_stack > ex20adj.tmp 2>&1; \
	  ${DIFF} output/ex20adj_2.out ex20adj.tmp || printf "${PWD}\nPossible problem with ex20adj_13, diffs above\n=========================================\n"; \
	  ${RM} -f ex20adj.tmp SA-data/*

runex20adj_14:
	-@${MPIEXEC} -n 1 ./ex20adj -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_stride 5 -ts_trajectory_solution_only -ts_trajectory_save_stack 0 > ex20adj.tmp 2>&1; \
	  ${DIFF} output/ex20adj_2.out ex20adj.tmp || printf "${PWD}\nPossible problem with ex20adj_14, diffs above\n=========================================\n"; \
	  ${RM} -f ex20adj.tmp SA-data/*

runex20adj_15:
	-@${MPIEXEC} -n 1 ./ex20adj -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_stride 5 -ts_trajectory_solution_only 0 -ts_trajectory_save_stack > ex20adj.tmp 2>&1; \
	  ${DIFF} output/ex20adj_2.out ex20adj.tmp || printf "${PWD}\nPossible problem with ex20adj_15, diffs above\n=========================================\n"; \
	  ${RM} -f ex20adj.tmp SA-data/*

runex20adj_16:
	-@${MPIEXEC} -n 1 ./ex20adj -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_stride 5 -ts_trajectory_solution_only 0 -ts_trajectory_save_stack 0 > ex20adj.tmp 2>&1; \
	  ${DIFF} output/ex20adj_2.out ex20adj.tmp || printf "${PWD}\nPossible problem with ex20adj_16, diffs above\n=========================================\n"; \
	  ${RM} -f ex20adj.tmp SA-data/*

runex20adj_17:
	-@${MPIEXEC} -n 1 ./ex20adj -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_max_cps_disk 8 -ts_trajectory_stride 5 -ts_trajectory_solution_only -ts_trajectory_save_stack > ex20adj.tmp 2>&1; \
	  ${DIFF} output/ex20adj_2.out ex20adj.tmp || printf "${PWD}\nPossible problem with ex20adj_17, diffs above\n=========================================\n"; \
	  ${RM} -f ex20adj.tmp SA-data/*

runex20adj_18:
	-@${MPIEXEC} -n 1 ./ex20adj -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_max_cps_disk 8 -ts_trajectory_stride 5 -ts_trajectory_solution_only 0 -ts_trajectory_save_stack > ex20adj.tmp 2>&1; \
	  ${DIFF} output/ex20adj_2.out ex20adj.tmp || printf "${PWD}\nPossible problem with ex20adj_18, diffs above\n=========================================\n"; \
	  ${RM} -f ex20adj.tmp SA-data/*

runex20adj_19:
	-@${MPIEXEC} -n 1 ./ex20adj -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_max_cps_disk 8 -ts_trajectory_stride 5 -ts_trajectory_solution_only -ts_trajectory_save_stack 0 > ex20adj.tmp 2>&1; \
	  ${DIFF} output/ex20adj_2.out ex20adj.tmp || printf "${PWD}\nPossible problem with ex20adj_19, diffs above\n=========================================\n"; \
	  ${RM} -f ex20adj.tmp SA-data/*

runex20adj_20:
	-@${MPIEXEC} -n 1 ./ex20adj -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_max_cps_disk 8 -ts_trajectory_stride 5 -ts_trajectory_solution_only 0 -ts_trajectory_save_stack 0 > ex20adj.tmp 2>&1; \
	  ${DIFF} output/ex20adj_2.out ex20adj.tmp || printf "${PWD}\nPossible problem with ex20adj_20, diffs above\n=========================================\n"; \
	  ${RM} -f ex20adj.tmp SA-data/*

runex20fwd:
	-@${MPIEXEC} -n 1 ./ex20fwd -monitor 0 -ts_type theta -ts_theta_endpoint -ts_theta_theta 0.5 > ex20fwd_1.tmp 2>&1; \
	  ${DIFF} output/ex20fwd_1.out ex20fwd_1.tmp || printf "${PWD}\nPossible problem with ex20fwd, diffs above\n=========================================\n"; \
	  ${RM} -f ex20fwd_1.tmp

runex20fwd_2:
	-@${MPIEXEC} -n 1 ./ex20fwd -monitor 0 -ts_type theta -ts_theta_endpoint -ts_theta_theta 0.5 -combined > ex20fwd_2.tmp 2>&1; \
	  ${DIFF} output/ex20fwd_1.out ex20fwd_2.tmp || printf "${PWD}\nPossible problem with ex20fwd, diffs above\n=========================================\n"; \
	  ${RM} -f ex20fwd_2.tmp

runex20opt_ic:
	-@${MPIEXEC} -n 1 ./ex20opt_ic -monitor 0 -ts_type theta -ts_theta_endpoint -ts_theta_theta 0.5 -viewer_binary_skip_info -tao_view -mu 1.0 > ex20opt_ic_1.tmp 2>&1; \
	  ${DIFF} output/ex20opt_ic_1.out ex20opt_ic_1.tmp || printf "${PWD}\nPossible problem with ex20opt_ic_1, diffs above\n=========================================\n"; \
	  ${RM} -f ex20opt_ic_1.tmp SA-data/*

runex20opt_p:
	-@${MPIEXEC} -n 1 ./ex20opt_p -monitor 0 -ts_type theta -ts_theta_endpoint -ts_theta_theta 0.5 -viewer_binary_skip_info -tao_view > ex20opt_p_1.tmp 2>&1; \
	  ${DIFF} output/ex20opt_p_1.out ex20opt_p_1.tmp || printf "${PWD}\nPossible problem with ex20opt_p_1, diffs above\n=========================================\n"; \
	  ${RM} -f ex20opt_p_1.tmp SA-data/*

runex22:
	-@${MPIEXEC} -n 1 ./ex22 -nox -da_grid_x 200 -ts_monitor_draw_solution -ts_arkimex_type 4 -ts_adapt_type none -ts_dt .005 -ts_final_time .1  > ex22_1.tmp 2>&1; \
	   ${DIFF} output/ex22_1.out ex22_1.tmp || printf "${PWD}\nPossible problem with ex22_1, diffs above\n=========================================\n";  \
	   ${RM} -f ex22_1.tmp
runex22_2:
	-@${MPIEXEC} -n 2 ./ex22 -nox -da_grid_x 200 -ts_monitor_draw_solution -ts_type rosw -ts_dt 1e-3 -ts_adapt_type none -ts_dt .005 -ts_final_time .1 > ex22_2.tmp 2>&1; \
	   ${DIFF} output/ex22_2.out ex22_2.tmp || printf "${PWD}\nPossible problem with ex22_2, diffs above\n=========================================\n";  \
	   ${RM} -f ex22_2.tmp
runex22_3:
	-@${MPIEXEC} -n 2 ./ex22 -nox -da_grid_x 200 -ts_monitor_draw_solution -ts_type rosw -ts_rosw_type ra34pw2 -ts_dt 5e-3 -ts_final_time .1  -ts_adapt_type none > ex22_3.tmp 2>&1; \
	   ${DIFF} output/ex22_3.out ex22_3.tmp || printf "${PWD}\nPossible problem with ex22_3, diffs above\n=========================================\n";  \
	   ${RM} -f ex22_3.tmp

runex22_4:
	-@${MPIEXEC} -n 2 ./ex22 -ts_type eimex -da_grid_x 200 -ts_eimex_order_adapt true -ts_dt 0.001 -ts_monitor -ts_max_steps 100 | ${SED} "s/ITS/TIME/g" > ex22_4.tmp 2>&1; \
	   ${DIFF} output/ex22_4.out ex22_4.tmp || printf "${PWD}\nPossible problem with ex22_4, diffs above\n=========================================\n";  \
	   ${RM} -f ex22_4.tmp

runex22f:
	-@${MPIEXEC} -n 1 ./ex22f -da_grid_x 200 -ts_arkimex_type 4 > ex22f_1.tmp 2>&1; \
	   ${DIFF} output/ex22f_1.out ex22f_1.tmp || printf "${PWD}\nPossible problem with ex22f_1, diffs above\n=========================================\n";  \
	   ${RM} -f ex22f_1.tmp

runex22f_mf:
	-@${MPIEXEC} -n 1 ./ex22f_mf -da_grid_x 200  -ts_arkimex_type 4 > ex22f_mf_1.tmp 2>&1; \
	   ${DIFF} output/ex22f_mf_1.out ex22f_mf_1.tmp || printf "${PWD}\nPossible problem with ex22f_mf_1, diffs above\n=========================================\n";  \
	   ${RM} -f ex22f_mf_1.tmp petscshiftmod.mod

runex24:                        # This is not in the nightlies because the convergence details are numerically sensitive
	-@${MPIEXEC} -n 1 ./ex24 -pc_type lu -ts_dt 1e-5 -ts_final_time 1e5 -n 50 -monitor_short > ex24_1.tmp 2>&1; \
	   ${DIFF} output/ex24_1.out ex24_1.tmp || printf "${PWD}\nPossible problem with ex24_1, diffs above\n=========================================\n";  \
	   ${RM} -f ex24_1.tmp

runextchem:
	-@${MPIEXEC} -n 1 ./extchem -Tini 1500 -ts_arkimex_fully_implicit -ts_max_snes_failures -1 -ts_adapt_monitor -ts_adapt_dt_max 1e-4 -ts_arkimex_type 4 -ts_final_time .005 > extchem_1.tmp 2>&1; \
	   ${DIFF} output/extchem_1.out extchem_1.tmp || printf "${PWD}\nPossible problem with extchem_1, diffs above\n=========================================\n";  \
	   ${RM} -f extchem_1.tmp

runex25:
	-@${MPIEXEC} -n 1 ./ex25 -nox -da_grid_x 20 -ts_monitor_draw_solution -ts_type rosw -ts_rosw_type 2p -ts_dt 5e-2 -ts_adapt_type none > ex25_1.tmp 2>&1; \
	   ${DIFF} output/ex25_1.out ex25_1.tmp || printf "${PWD}\nPossible problem with ex25_1, diffs above\n=========================================\n";  \
	   ${RM} -f ex25_1.tmp

runex26:
	-@${MPIEXEC} -n 1 ./ex26 -da_grid_x 20 -da_grid_y 20 -lidvelocity 100 -grashof 1e3 -ts_max_steps 100 -ts_rtol 1e-3 -ts_atol 1e-3 -ts_type rosw -ts_rosw_type ra3pw -ts_monitor -ts_monitor_solution_vtk 'foo-%03D.vts' > ex26_1.tmp 2>&1; \
	   ${DIFF} output/ex26_1.out ex26_1.tmp || printf "${PWD}\nPossible problem with ex26_1, diffs above\n=========================================\n";  \
	   ${RM} -f ex26_1.tmp

runex26_2:
	-@${MPIEXEC} -n 4 ./ex26 -da_refine 2 -lidvelocity 100 -grashof 1e3 -ts_max_steps 10 -ts_rtol 1e-3 -ts_atol 1e-3 -pc_type lu -ts_type beuler -ts_monitor -snes_monitor_short -snes_type ngmres -npc_snes_type nasm -npc_snes_nasm_type restrict -da_overlap 4  > ex26_2.tmp 2>&1; \
	   ${DIFF} output/ex26_2.out ex26_2.tmp || printf "${PWD}\nPossible problem with ex26_2, diffs above\n=========================================\n";  \
	   ${RM} -f ex26_2.tmp

runex26_3:
	-@${MPIEXEC} -n 4 ./ex26 -da_refine 2 -lidvelocity 100 -grashof 1e3 -ts_max_steps 10 -ts_rtol 1e-3 -ts_atol 1e-3 -pc_type lu -ts_type beuler -ts_monitor -snes_monitor_short -snes_type aspin -da_overlap 4  > ex26_3.tmp 2>&1; \
	   ${DIFF} output/ex26_3.out ex26_3.tmp || printf "${PWD}\nPossible problem with ex26_3, diffs above\n=========================================\n";  \
	   ${RM} -f ex26_3.tmp

runex26_4:
	-@${MPIEXEC} -n 2 ./ex26 -da_refine 1 -lidvelocity 100 -grashof 1e3 -ts_max_steps 10 -ts_rtol 1e-3 -ts_atol 1e-3 \
        -ts_type beuler -ts_monitor -snes_monitor_short -snes_fd_color -pc_type mg  > ex26_4.tmp 2>&1; \
	   ${DIFF} output/ex26_4.out ex26_4.tmp || printf "${PWD}\nPossible problem with ex26_4, diffs above\n=========================================\n";  \
	   ${RM} -f ex26_4.tmp

runex26_asm:
	-@${MPIEXEC} -n 4 ./ex26 -da_refine 1 -lidvelocity 100 -grashof 1e3 -ts_max_steps 10 -ts_rtol 1e-3 -ts_atol 1e-3 \
        -ts_type beuler -ts_monitor -snes_converged_reason -pc_type asm > ex26_4.tmp 2>&1; \
	   ${DIFF} output/ex26_asm.out ex26_4.tmp || printf "${PWD}\nPossible problem with ex26_asm, diffs above\n=========================================\n";  \
	   ${RM} -f ex26_4.tmp

runex29:
	-@${MPIEXEC} -n 16 ./ex29 -da_refine 4 -ts_max_steps 10 -ts_rtol 1e-3 -ts_atol 1e-3 \
        -ts_type arkimex -ts_monitor -snes_monitor -snes_type ngmres \
        -npc_snes_type nasm -npc_snes_nasm_type restrict -da_overlap 4  > ex29.tmp 2>&1; \
	   ${DIFF} output/ex29.out ex29.tmp || printf "${PWD}\nPossible problem with ex29, diffs above\n=========================================\n";  \
	   ${RM} -f ex29.tmp

runex31:
	-@${MPIEXEC} -n 1 ./ex31 > ex31_1.tmp 2>&1; \
	   ${DIFF} output/ex31_1.out ex31_1.tmp || printf "${PWD}\nPossible problem with ex31_1, diffs above\n=========================================\n";  \
	   ${RM} -f ex31_1.tmp

runex31_2:
	-@${MPIEXEC} -n 1 ./ex31 -ts_type glee > ex31_2.tmp 2>&1; \
	   ${DIFF} output/ex31_2.out ex31_2.tmp || printf "${PWD}\nPossible problem with ex31_2, diffs above\n=========================================\n";  \
	   ${RM} -f ex31_2.tmp

runex35:
	-@${MPIEXEC} -n 1 ./ex35 -n 20 -ts_type rosw -ts_rosw_type 2p -ts_dt 5e-2 -ts_adapt_type none  > ex35_1.tmp 2>&1;	  \
	   ${DIFF} output/ex35_1.out ex35_1.tmp || printf "${PWD}\nPossible problem with ex35_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex35_1.tmp

runex35_2:
	-@${MPIEXEC} -n 2 ./ex35 -n 50 -ts_type gl -ts_adapt_type none -ts_dt 0.1 -io > ex35_2.tmp 2>&1;	  \
	   ${DIFF} output/ex35_2.out ex35_2.tmp || printf "${PWD}\nPossible problem with ex35_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex35_2.tmp

runex40: runex40_a runex40_b runex40_c runex40_d runex40_e

runex40_a:
	-@${MPIEXEC} -n 1 ./ex40 -snes_stol 1e-4 > ex40.tmp 2>&1; \
	   ${DIFF} output/ex40.out ex40.tmp || printf "${PWD}\nPossible problem with ex40_a, diffs above\n=========================================\n"; \
	   ${RM} -f ex40.tmp SA-data/*

runex40_b:
	-@${MPIEXEC} -n 1 ./ex40 -ts_type arkimex -snes_stol 1e-4 > ex40.tmp 2>&1; \
	   ${DIFF} output/ex40.out ex40.tmp || printf "${PWD}\nPossible problem with ex40_b, diffs above\n=========================================\n"; \
	   ${RM} -f ex40.tmp SA-data/*

runex40_c:
	-@${MPIEXEC} -n 1 ./ex40 -ts_type theta -ts_adapt_type basic -ts_atol 1e-1 -snes_stol 1e-4 > ex40.tmp 2>&1; \
	   ${DIFF} output/ex40.out ex40.tmp || printf "${PWD}\nPossible problem with ex40_c, diffs above\n=========================================\n"; \
	   ${RM} -f ex40.tmp SA-data/*

runex40_d:
	-@${MPIEXEC} -n 1 ./ex40 -ts_type alpha -ts_adapt_type basic -ts_atol 1e-1 -snes_stol 1e-4 > ex40.tmp 2>&1; \
	   ${DIFF} output/ex40.out ex40.tmp || printf "${PWD}\nPossible problem with ex40_d, diffs above\n=========================================\n"; \
	   ${RM} -f ex40.tmp SA-data/*

runex40_e:
	-@${MPIEXEC} -n 1 ./ex40 -ts_type bdf -ts_adapt_dt_max 0.025 -ts_max_steps 1500 > ex40.tmp 2>&1; \
	   ${DIFF} output/ex40.out ex40.tmp || printf "${PWD}\nPossible problem with ex40_e, diffs above\n=========================================\n"; \
	   ${RM} -f ex40.tmp SA-data/*

runex40_f:
	-@${MPIEXEC} -n 1 ./ex40 -rhs-form -ts_type rk -ts_rk_type 3bs > ex40.tmp 2>&1; \
	   ${DIFF} output/ex40.out ex40.tmp || printf "${PWD}\nPossible problem with ex40_f, diffs above\n=========================================\n"; \
	   ${RM} -f ex40.tmp SA-data/*

runex40_g:
	-@${MPIEXEC} -n 1 ./ex40 -rhs-form -ts_type rk -ts_rk_type 5bs > ex40.tmp 2>&1; \
	   ${DIFF} output/ex40.out ex40.tmp || printf "${PWD}\nPossible problem with ex40_g, diffs above\n=========================================\n"; \
	   ${RM} -f ex40.tmp SA-data/*

runex42:
	-@${MPIEXEC} -n 1 ./ex42 -ts_max_steps 8  > ex42.tmp 2>&1;  \
	   ${DIFF} output/ex42.out ex42.tmp || printf "${PWD}\nPossible problem with ex42, diffs above\n=========================================\n"; \
	   ${RM} -f ex42.tmp

runex43_a:
	-@${MPIEXEC} -n 1 ./ex43 -ts_max_steps 10 -ts_view > ex43_a.tmp 2>&1; \
	   ${DIFF} output/ex43_a.out ex43_a.tmp || printf "${PWD}\nPossible problem with ex43_a, diffs above\n=========================================\n"; \
	   ${RM} -f ex43_a.tmp

runex43_b:
	-@${MPIEXEC} -n 1 ./ex43 -ts_max_steps 10 -ts_rtol 0 -ts_atol 1e-5 -ts_adapt_type basic -ts_adapt_monitor > ex43_b.tmp 2>&1; \
	   ${DIFF} output/ex43_b.out ex43_b.tmp || printf "${PWD}\nPossible problem with ex43_b, diffs above\n=========================================\n"; \
	   ${RM} -f ex43_b.tmp

runex44_a:
	-@${MPIEXEC} -n 1 ./ex44 > ex44_a.tmp 2>&1; \
	   ${DIFF} output/ex44.out ex44_a.tmp || printf "${PWD}\nPossible problem with ex44_a, diffs above\n=========================================\n"; \
	   ${RM} -f ex44_a.tmp

runex44_b:
	-@${MPIEXEC} -n 1 ./ex44 -ts_rtol 0 -ts_atol 1e-1 -ts_adapt_type basic > ex44_b.tmp 2>&1; \
	   ${DIFF} output/ex44.out ex44_b.tmp || printf "${PWD}\nPossible problem with ex44_b, diffs above\n=========================================\n"; \
	   ${RM} -f ex44_b.tmp

runex44_2:
	-@${MPIEXEC} -n 2 ./ex44 | sort -b > ex44_2.tmp 2>&1; \
	   sort  -b output/ex44_2.out | ${DIFF}  - ex44_2.tmp || printf "${PWD}\nPossible problem with ex44_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex44_2.tmp

TESTEXAMPLES_C = ex1.PETSc runex1 runex1_2 ex1.rm ex3.PETSc runex3 runex3_2 runex3_4 runex3_5 ex3.rm \
                            ex4.PETSc runex4 runex4_2 runex4_3 runex4_4 ex4.rm ex5.PETSc runex5_nox ex5.rm\
                            ex6.PETSc runex6 ex6.rm ex7.PETSc runex7 runex7_2 runex7_3 ex7.rm \
                            ex8.PETSc runex8 ex8.rm \
                            ex13.PETSc runex13 runex13_2 runex13_3 ex13.rm\
                            ex15.PETSc runex15 runex15_2 runex15_5 ex15.rm \
                            ex16adj.PETSc runex16adj runex16adj_2 ex16adj.rm printdot \
                            ex17.PETSc runex17 ex17.rm \
                            ex19.PETSc ex19.rm \
                            ex22.PETSc runex22_2 runex22_3 runex22_4 ex22.rm \
                            ex25.PETSc runex25 ex25.rm \
                            ex31.PETSc runex31 runex31_2 ex31.rm \
                            ex40.PETSc runex40_a runex40_b runex40_c runex40_d runex40_e runex40_f runex40_g ex40.rm \
                            ex43.PETSc ex43.rm \
                            ex44.PETSc runex44_a runex44_b runex44_2 ex44.rm

TESTEXAMPLES_C_NOTSINGLE = ex2.PETSc runex2 ex2.rm ex3.PETSc runex3_3 ex3.rm ex8.PETSc runex8_2 runex8_3 ex8.rm ex10.PETSc printdot runex10 runex10_2 runex10_3  ex10.rm \
                            printdot ex12.PETSc runex12 runex12_2 ex12.rm ex15.PETSc runex15_3 runex15_4 ex15.rm printdot \
                            ex16.PETSc runex16  ex16.rm ex17.PETSc runex17 runex17_2 ex17.rm  ex22.PETSc runex22  ex22.rm printdot \
                            ex29.PETSc runex29 ex29.rm ex43.PETSc runex43_a runex43_b ex43.rm
TESTEXAMPLES_C_NOCOMPLEX_NOTSINGLE  = ex9.PETSc runex9 runex9_2 runex9_3 ex9.rm \
                            ex26.PETSc runex26 runex26_2 runex26_3 runex26_4 runex26_asm ex26.rm\
                            ex16opt_p.PETSc  runex16opt_p  ex16opt_p.rm \
                            printdot ex16opt_ic.PETSc runex16opt_ic ex16opt_ic.rm \
                            ex20fwd.PETSc runex20fwd runex20fwd_2 ex20fwd.rm \
                            ex20opt_p.PETSc runex20opt_p ex20opt_p.rm \
                            ex20opt_ic.PETSc runex20opt_ic ex20opt_ic.rm
TESTEXAMPLES_C_X = ex5.PETSc runex5 ex5.rm
TESTEXAMPLES_FORTRAN = ex1f.PETSc runex1f ex1f.rm
TESTEXAMPLES_FORTRAN_NOTSINGLE = ex22f.PETSc runex22f ex22f.rm
TESTEXAMPLES_F90_NOTSINGLE = ex22f_mf.PETSc runex22f_mf ex22f_mf.rm
TESTEXAMPLES_C_X_MPIUNI = ex40.PETSc runex40_a runex40_b runex40_c runex40_d runex40_e runex40_f runex40_g ex40.rm \
                            ex43.PETSc runex43_a runex43_b ex43.rm \
                            ex44.PETSc runex44_a runex44_b ex44.rm
TESTEXAMPLES_13 = ex2.PETSc ex2.rm ex3.PETSc ex3.rm ex4.PETSc ex4.rm \
                            ex5.PETSc ex5.rm
TESTEXAMPLES_EXODUSII = ex11.PETSc runex11 ex11.rm
TESTEXAMPLES_MOAB = ex35.PETSc runex35 ex35.rm
TESTEXAMPLES_TCHEM = extchem.PETSc runextchem extchem.rm

TESTEXAMPLES_REVOLVE = ex20adj.PETSc runex20adj_7 runex20adj_8 runex20adj_9 printdot runex20adj_10 runex20adj_11 runex20adj_12 runex20adj_13 runex20adj_14 runex20adj_15 runex20adj_16 runex20adj_17 printdot runex20adj_18 runex20adj_19 runex20adj_20 ex20adj.rm

include ${PETSC_DIR}/lib/petsc/conf/test
