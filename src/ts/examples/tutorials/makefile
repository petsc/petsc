
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

TESTEXAMPLES_C =                                 ex22.PETSc runex22_2 runex22_3 runex22_4 ex22.rm \
                            ex25.PETSc runex25 ex25.rm \
                            ex31.PETSc runex31 runex31_2 ex31.rm \
                            ex40.PETSc runex40_a runex40_b runex40_c runex40_d runex40_e runex40_f runex40_g ex40.rm \
                            ex43.PETSc ex43.rm \
                            ex44.PETSc runex44_a runex44_b runex44_2 ex44.rm

TESTEXAMPLES_C_NOTSINGLE =  ex22.PETSc runex22  ex22.rm printdot \
                            ex29.PETSc runex29 ex29.rm ex43.PETSc runex43_a runex43_b ex43.rm
TESTEXAMPLES_C_NOCOMPLEX_NOTSINGLE  =                            ex26.PETSc runex26 runex26_2 runex26_3 runex26_4 runex26_asm ex26.rm\
                            ex20fwd.PETSc runex20fwd runex20fwd_2 ex20fwd.rm \
                            ex20opt_p.PETSc runex20opt_p ex20opt_p.rm \
                            ex20opt_ic.PETSc runex20opt_ic ex20opt_ic.rm
TESTEXAMPLES_FORTRAN_NOTSINGLE = ex22f.PETSc runex22f ex22f.rm
TESTEXAMPLES_F90_NOTSINGLE = ex22f_mf.PETSc runex22f_mf ex22f_mf.rm
TESTEXAMPLES_C_X_MPIUNI = ex40.PETSc runex40_a runex40_b runex40_c runex40_d runex40_e runex40_f runex40_g ex40.rm \
                            ex43.PETSc runex43_a runex43_b ex43.rm \
                            ex44.PETSc runex44_a runex44_b ex44.rm
TESTEXAMPLES_13 =                             ex5.PETSc ex5.rm
TESTEXAMPLES_MOAB = ex35.PETSc runex35 ex35.rm
TESTEXAMPLES_TCHEM = extchem.PETSc runextchem extchem.rm


include ${PETSC_DIR}/lib/petsc/conf/test
