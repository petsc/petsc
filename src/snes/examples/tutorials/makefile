
# This directory contains SNES example programs for solving systems of
# nonlinear equations.  

#CPPFLAGS	 = -I/PETSc3/geodynamics/PetscSimulationsViewers/src
CFLAGS           =
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/snes/examples/tutorials/
MANSEC           = SNES
EXAMPLESC	 = ex1.c ex2.c ex3.c ex4.c ex4tu.c ex5.c ex5s.c ex6.c ex7.c ex8.c \
                ex10.c ex12.c ex14.c ex18.c ex19.c ex19tu.c ex20.c ex21.c ex22.c \
                ex23.c ex24.c ex25.c ex26.c ex27.c ex28.c ex29.c ex30.c ex31.c \
                ex32.c ex38.c ex42.c ex43.c ex44.c ex45.c ex46.c ex48.c ex49.c \
                ex50.c ex51.c ex52.c ex53.c ex54.c ex55.c ex58.c ex59.c ex60.c \
                ex61.c ex61gen.c ex61view.c ex63.c ex64.c ex65.c
EXAMPLESF	 = ex1f.F ex5f.F ex5f90.F ex5f90t.F ex5fs.F ex39f90.F ex40f90.F
EXAMPLESCH	 = ex43-44.h
EXAMPLESFH       = ex5f.h
EXAMPLESMATLAB   = ex5m.m ex29view.m ex61genm.m ex61m.m
DIRS		 = ex10d

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_SNES_LIB}
	${RM} ex1.o
ex1f: ex1f.o  chkopts
	-${FLINKER} -o ex1f ex1f.o ${PETSC_SNES_LIB}
	${RM} ex1f.o
ex1f90: ex1f90.o  chkopts
	-${FLINKER} -o ex1f90 ex1f90.o ${PETSC_SNES_LIB}
	${RM} ex1f90.o
ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_SNES_LIB}
	${RM} ex2.o
ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o ${PETSC_SNES_LIB}
	${RM} ex3.o
ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o ${PETSC_SNES_LIB}
	${RM} ex4.o
ex4tu: ex4tu.o  chkopts
	-${CLINKER} -o ex4tu ex4tu.o ${PETSC_SNES_LIB}
	${RM} ex4tu.o
ex5: ex5.o chkopts
	-${CLINKER} -o ex5 ex5.o ${PETSC_SNES_LIB}
	${RM} ex5.o
ex5f: ex5f.o  chkopts
	-${FLINKER} -o ex5f ex5f.o ${PETSC_SNES_LIB}
	${RM} ex5f.o
#
#  The SGI parallelizing compiler generates incorrect code by treating 
#  the math functions (such as sqrt and exp) as local variables. The 
#  sed below patches this.
#
ex5s: chkopts
	@if [ "${PETSC_ARCH}" != "IRIX64" ]; then echo "Only for PETSC_ARCH of IRIX64"; false ; fi
	-${CC} -pca keep  -WK,-lo=l ${FCONF} ${CFLAGS} -c ex5s.c
	sed "s/, sqrt/ /g"   ex5s.M | sed "s/, exp/ /g"  > ex5s_tmp.c
	-${CC} -mp ${PCC_FLAGS} ${CFLAGS} ${CCPPFLAGS} -c ex5s_tmp.c
	-${FC} -pfa keep -mp -64 ${FC_FLAGS} ${FFLAGS} ${FCPPFLAGS} -c ex5fs.F
	-${CLINKER} -mp -o ex5s ex5s_tmp.o ex5fs.o ${PETSC_SNES_LIB}
	${RM} ex5s.o
ex5f90: ex5f90.o  chkopts
	-${FLINKER} -o ex5f90 ex5f90.o ${PETSC_SNES_LIB}
	${RM} ex5f90.o
ex5f90t: ex5f90t.o  chkopts
	-${FLINKER} -o ex5f90t ex5f90t.o ${PETSC_SNES_LIB}
	${RM} ex5f90t.o
ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o ${PETSC_SNES_LIB}
	${RM} ex6.o
ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o -lsimviewers ${PETSC_SNES_LIB}
	${RM} ex7.o
ex8: ex8.o chkopts
	-${CLINKER} -o ex8 ex8.o -lsimviewers ${PETSC_SNES_LIB}
	${RM} ex8.o
ex9: ex9.o chkopts
	-${CLINKER} -o ex9 ex9.o ${PETSC_SNES_LIB}
	${RM} ex9.o
ex10: ex10.o chkopts
	-${CLINKER} -o ex10 ex10.o ${PETSC_SNES_LIB}
	${RM} ex10.o
ex12: ex12.o chkopts
	-${CLINKER} -o ex12 ex12.o ${PETSC_SNES_LIB}
	${RM} ex12.o
ex13: ex13.o chkopts
	-${CLINKER} -o ex13 ex13.o ${PETSC_SNES_LIB}
	${RM} ex13.o
ex14: ex14.o chkopts
	-${CLINKER} -o ex14 ex14.o ${PETSC_SNES_LIB}
	${RM} ex14.o
ex15: ex15.o chkopts
	-${CLINKER} -o ex15 ex15.o ${PETSC_SNES_LIB}
	${RM} ex15.o
ex16: ex16.o chkopts
	-${CLINKER} -o ex16 ex16.o ${PETSC_SNES_LIB}
	${RM} ex16.o
ex17: ex17.o chkopts
	-${CLINKER} -o ex17 ex17.o ${PETSC_SNES_LIB}
	${RM} ex17.o
ex18: ex18.o chkopts
	-${CLINKER} -o ex18 ex18.o ${PETSC_SNES_LIB}
	${RM} ex18.o
ex19:  ex19.o chkopts
	-${CLINKER} -o ex19  ex19.o ${PETSC_SNES_LIB}
	${RM} ex19.o
ex19tu:  ex19tu.o chkopts
	-${CLINKER} -o ex19tu  ex19tu.o ${PETSC_SNES_LIB}
	${RM} ex19tu.o
ex20: ex20.o chkopts
	-${CLINKER} -o ex20 ex20.o ${PETSC_SNES_LIB}
	${RM} ex20.o
ex21: ex21.o chkopts
	-${CLINKER} -o ex21 ex21.o ${PETSC_SNES_LIB}
	${RM} ex21.o
ex22: ex22.o chkopts
	-${CLINKER} -o ex22 ex22.o ${PETSC_SNES_LIB}
	${RM} ex22.o
ex23: ex23.o chkopts
	-${CLINKER} -o ex23 ex23.o ${PETSC_SNES_LIB}
	${RM} ex23.o
ex24: ex24.o chkopts
	-${CLINKER} -o ex24 ex24.o ${PETSC_SNES_LIB}
	${RM} ex24.o
ex25: ex25.o chkopts
	-${CLINKER} -o ex25 ex25.o ${PETSC_SNES_LIB}
	${RM} ex25.o
ex26: ex26.o chkopts
	-${CLINKER} -o ex26 ex26.o ${PETSC_SNES_LIB}
	${RM} ex26.o
ex27: ex27.o chkopts
	-${CLINKER} -o ex27 ex27.o ${PETSC_SNES_LIB}
	${RM} ex27.o
ex28: ex28.o chkopts
	-${CLINKER} -o ex28 ex28.o ${PETSC_SNES_LIB}
	${RM} ex28.o
ex29: ex29.o chkopts
	-${CLINKER} -o ex29 ex29.o ${PETSC_SNES_LIB}
	${RM} ex29.o
ex30: ex30.o chkopts
	-${CLINKER} -o ex30 ex30.o ${PETSC_SNES_LIB}
	${RM} ex30.o
ex31: ex31.o chkopts
	-${CLINKER} -o ex31 ex31.o ${PETSC_SNES_LIB}
	${RM} ex31.o
ex32: ex32.o chkopts
	-${CLINKER} -o ex32 ex32.o ${PETSC_SNES_LIB}
	${RM} ex32.o
ex38: ex38.o  chkopts
	-${CLINKER} -o ex38 ex38.o ${PETSC_SNES_LIB}
	${RM} ex38.o
ex39f90: ex39f90.o  chkopts
	-${FLINKER} -o ex39f90 ex39f90.o ${PETSC_SNES_LIB}
	${RM} ex39f90.o
ex40f90: ex40f90.o  chkopts
	-${FLINKER} -o ex40f90 ex40f90.o ${PETSC_SNES_LIB}
	${RM} ex40f90.o
ex41: ex41.o  chkopts
	-${CLINKER} -o ex41 ex41.o ${PETSC_SNES_LIB}
	${RM} ex41.o
ex42: ex42.o  chkopts
	-${CLINKER} -o ex42 ex42.o ${PETSC_SNES_LIB}
	${RM} ex42.o
ex43: ex43.o  chkopts
	-${CLINKER} -o ex43 ex43.o ${PETSC_SNES_LIB}
	${RM} ex43.o
ex44: ex44.o  chkopts
	-${CLINKER} -o ex44 ex44.o ${PETSC_SNES_LIB}
	${RM} ex44.o
ex45: ex45.o  chkopts
	-${CLINKER} -o ex45 ex45.o ${PETSC_SNES_LIB}
	${RM} ex45.o
ex46: ex46.o  chkopts
	-${CLINKER} -o ex46 ex46.o ${PETSC_SNES_LIB}
	${RM} ex46.o
ex47cu: ex47cu.o  chkopts
	-${CLINKER} -o ex47cu ex47cu.o ${PETSC_SNES_LIB}
	${RM} ex47cu.o
ex48: ex48.o  chkopts
	-${CLINKER} -o ex48 ex48.o ${PETSC_SNES_LIB}
	${RM} ex48.o
ex49: ex49.o  chkopts
	-${CLINKER} -o ex49 ex49.o ${PETSC_SNES_LIB}
	${RM} ex49.o
ex50: ex50.o  chkopts
	-${CLINKER} -o ex50 ex50.o ${PETSC_SNES_LIB}
	${RM} ex50.o
ex51: ex51.o  chkopts
	-${CLINKER} -o ex51 ex51.o ${PETSC_SNES_LIB}
	${RM} ex51.o
ex52: ex52.o  chkopts
	-${CLINKER} -o ex52 ex52.o ${PETSC_SNES_SINGLE_LIB}
	${RM} ex52.o
ex53: ex53.o  chkopts
	-${CLINKER} -o ex53 ex53.o ${PETSC_SNES_LIB}
	${RM} ex53.o
ex54: ex54.o  chkopts
	-${CLINKER} -o ex54 ex54.o ${PETSC_SNES_LIB}
	${RM} ex54.o
ex55: ex55.o  chkopts
	-${CLINKER} -o ex55 ex55.o ${PETSC_SNES_LIB}
	${RM} ex55.o
ex56: ex56.o  chkopts
	-${CLINKER} -o ex56 ex56.o ${PETSC_SNES_LIB}
	${RM} ex56.o
ex57: ex57.o  chkopts
	-${CLINKER} -o ex57 ex57.o ${PETSC_SNES_LIB}
	${RM} ex57.o
ex58: ex58.o  chkopts
	-${CLINKER} -o ex58 ex58.o ${PETSC_SNES_LIB}
	${RM} ex58.o
ex59: ex59.o  chkopts
	-${CLINKER} -o ex59 ex59.o ${PETSC_SNES_LIB}
	${RM} ex59.o
ex60: ex60.o  chkopts
	-${CLINKER} -o ex60 ex60.o ${PETSC_SNES_LIB}
	${RM} ex60.o
ex61: ex61.o  chkopts
	-${CLINKER} -o ex61 ex61.o ${PETSC_SNES_LIB}
	${RM} ex61.o
ex61gen: ex61gen.o  chkopts
	-${CLINKER} -o ex61gen ex61gen.o ${PETSC_SNES_LIB}
	${RM} ex61gen.o
ex61view: ex61view.o  chkopts
	-${CLINKER} -o ex61view ex61view.o ${PETSC_SNES_LIB}
	${RM} ex61view.o
ex62: ex62.o  chkopts
	-${CLINKER} -o ex62 ex62.o ${PETSC_SNES_LIB}
	${RM} ex62.o
ex63: ex63.o  chkopts
	-${CLINKER} -o ex63 ex63.o ${PETSC_SNES_LIB}
	${RM} ex63.o
ex64: ex64.o  chkopts
	-${CLINKER} -o ex64 ex64.o ${PETSC_SNES_LIB}
	${RM} ex64.o
ex65: ex65.o  chkopts
	-${CLINKER} -o ex65 ex65.o ${PETSC_SNES_LIB}
	${RM} ex65.o
#--------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 -ksp_gmres_cgs_refinement_type refine_always -snes_monitor_short > ex1_1.tmp 2>&1;   \
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   ${RM} -f ex1_1.tmp
runex1_2:
	-@${MPIEXEC} -n 2 ./ex1 -snes_mf_operator -snes_fd -snes_monitor_short > ex1_2.tmp 2>&1;   \
	   if (${DIFF} output/ex1_2.out ex1_2.tmp) then true; \
	   else echo "Possible problem with ex1_2, diffs above"; fi; \
	   ${RM} -f ex1_2.tmp
runex1f:
	-@${MPIEXEC} -n 1 ./ex1f -ksp_gmres_cgs_refinement_type refine_always -snes_monitor_short > ex1f_1.tmp 2>&1;   \
	   if (${DIFF} output/ex1f_1.out ex1f_1.tmp) then true; \
	   else echo "Possible problem with ex1f_1, diffs above"; fi; \
	   ${RM} -f ex1f_1.tmp
runex2:
	-@${MPIEXEC} -n 1 ./ex2 -nox -snes_monitor_cancel -snes_monitor_short \
		 -snes_view -pc_type jacobi -ksp_gmres_cgs_refinement_type refine_always >ex2_1.tmp 2>&1;	\
	   if (${DIFF} output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi; \
	   ${RM} -f ex2_1.tmp
runex2_2:
	-@${MPIEXEC} -n 1 ./ex2 -nox -snes_monitor_cancel -snes_monitor_short \
		-snes_type tr -snes_view \
		-pc_type jacobi -ksp_gmres_cgs_refinement_type refine_always > ex2_2.tmp 2>&1; \
	   if (${DIFF} output/ex2_2.out ex2_2.tmp) then true; \
	   else echo "Possible problem with ex2_2, diffs above"; fi; \
	   ${RM} -f ex2_2.tmp
runex2_3:
	-@${MPIEXEC} -n 1 ./ex2 -nox -snes_monitor_cancel -snes_monitor_short -malloc no \
		 -snes_view -pc_type jacobi -ksp_gmres_cgs_refinement_type refine_always >ex2_3.tmp 2>&1;	\
	   if (${DIFF} output/ex2_1.out ex2_3.tmp) then true; \
	   else echo "Possible problem with ex2_3, diffs above"; fi; \
	   ${RM} -f ex2_3.tmp
runex3:
	-@${MPIEXEC} -n 1 ./ex3 -nox -snes_monitor_cancel -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex3_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex3_1.out ex3_1.tmp) then true; \
	   else echo "Possible problem with ex3_1, diffs above"; fi; \
	   ${RM} -f ex3_1.tmp
runex3_2:
	-@${MPIEXEC} -n 3 ./ex3 -nox -pc_type asm -mat_type mpiaij  \
	   -snes_monitor_cancel -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex3_2.tmp 2>&1; \
	   if (${DIFF} output/ex3_2.out ex3_2.tmp) then true; \
	   else echo "Possible problem with ex3_2, diffs above"; fi; \
	   ${RM} -f ex3_2.tmp
runex3_3:
	-@${MPIEXEC} -n 2 ./ex3 -nox -snes_monitor_cancel -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex3_3.tmp 2>&1; \
	   if (${DIFF} output/ex3_3.out ex3_3.tmp) then true; \
	   else echo "Possible problem with ex3_3, diffs above"; fi; \
	   ${RM} -f ex3_3.tmp
runex3_4:
	-@${MPIEXEC} -n 1 ./ex3 -nox -pre_check_iterates -post_check_iterates > ex3_4.tmp 2>&1; \
	   if (${DIFF} output/ex3_4.out ex3_4.tmp) then true; \
	   else echo "Possible problem with ex3_4, diffs above"; fi; \
	   ${RM} -f ex3_4.tmp
runex5:
	-@${MPIEXEC} -n 1 ./ex5 -pc_type mg -ksp_monitor_short  -snes_view -pc_mg_levels 3 -pc_mg_galerkin -da_grid_x 17 -da_grid_y 17 \
                    -mg_levels_ksp_monitor_short -snes_monitor_short -mg_levels_pc_type sor -pc_mg_type full  > ex5_1.tmp 2>&1; \
	   if (${DIFF} output/ex5_1.out ex5_1.tmp) then true; \
	   else echo "Possible problem with ex5, diffs above"; fi; \
	   ${RM} -f ex5_1.tmp
runex5_2:
	-@${MPIEXEC} -n 1 ./ex5   -pc_type mg -ksp_monitor_short  -snes_view -pc_mg_galerkin -snes_grid_sequence 3 \
                           -mg_levels_ksp_monitor_short -snes_monitor_short -mg_levels_pc_type sor -pc_mg_type full  > ex5_2.tmp 2>&1; \
	   if (${DIFF} output/ex5_2.out ex5_2.tmp) then true; \
	   else echo "Possible problem with ex5_2, diffs above"; fi; \
	   ${RM} -f ex5_2.tmp
runex5_3:
	-@${MPIEXEC} -n 2 ./ex5 -snes_grid_sequence 2 -snes_monitor_short -ksp_monitor_short -snes_mf_operator -ksp_converged_reason \
           -snes_converged_reason -snes_view -pc_type mg    > ex5_3.tmp 2>&1; \
	   if (${DIFF} output/ex5_3.out ex5_3.tmp) then true; \
	   else echo "Possible problem with ex5_3, diffs above"; fi; \
	   ${RM} -f ex5_3.tmp
runex5_4:
	-@${MPIEXEC} -n 2 ./ex5 -snes_grid_sequence 2 -snes_monitor_short -ksp_monitor_short -ksp_converged_reason \
           -snes_converged_reason -snes_view -pc_type mg    > ex5_4.tmp 2>&1; \
	   if (${DIFF} output/ex5_4.out ex5_4.tmp) then true; \
	   else echo "Possible problem with ex5_4, diffs above"; fi; \
	   ${RM} -f ex5_4.tmp

runex5_5_ngmres:
	-@${MPIEXEC} -n 2 ./ex5 -da_grid_x 20 -da_grid_y 20 \
	-snes_type ngmres -npc_snes_picard_alpha 0.1 -npc_snes_picard quadratic -snes_monitor_short -snes_max_it 1000	 \
	-npc_snes_type picard -npc_snes_max_it 1 -snes_ngmres_m 30 -snes_ngmres_restart 1000 -snes_ngmres_debug 0 	\
	-par 6.0
runex5_5_picard:
	-@${MPIEXEC} -n 2 ./ex5 -da_grid_x 20 -da_grid_y 20 \
	-snes_type picard -snes_picard_alpha 0.1 -snes_picard quadratic -snes_monitor_short -snes_max_it 10000 \
	-par 6.0 
runex5_5_sorqn:
	-@${MPIEXEC} -n 2 ./ex5 -da_grid_x 10 -da_grid_y 10 \
	-snes_type sorqn -snes_sorqn_jacobian_start 0 -snes_sorqn_restart 10 -snes_sorqn_alpha 0.25 -snes_monitor -snes_max_it 10000 \
	-par 6.0 

runex5_5_ngmres_sorqn:
	-@${MPIEXEC} -n 2 ./ex5 -da_grid_x 20 -da_grid_y 20 \
	-snes_type ngmres -snes_ngmres_m 30 -snes_ngmres_restart 1000 -snes_monitor_short -snes_max_it 1000	 \
	-npc_snes_type sorqn -npc_snes_sorqn_jacobian_start 0 -npc_snes_max_it 3 -npc_snes_sorqn_alpha 0.25 -npc_snes_monitor_short \
	-par 6.0

runex5_6:
	-@${MPIEXEC} -n 4 ./ex5 -snes_monitor_short -ksp_monitor_short -da_grid_x 129 -da_grid_y 129  -pc_type mg -pc_mg_levels 8 -mg_levels_ksp_type chebychev -mg_levels_ksp_chebychev_estimate_eigenvalues 0,0.5,0,1.1 -mg_levels_ksp_max_it 2 > ex5_6.tmp 2>&1; \
	   ${DIFF} output/ex5_6.out ex5_6.tmp || echo "Possible problem with ex5_6, diffs above"; \
	   ${RM} -f ex5_6.tmp

runex5f:
	-@${MPIEXEC} -n 4 ./ex5f -snes_mf -da_processors_x 4 -da_processors_y 1 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5f_1.tmp 2>&1; \
	   if (${DIFF} output/ex5f_1.out ex5f_1.tmp) then true; \
	   else echo "Possible problem with ex5f_1, diffs above"; fi; \
	   ${RM} -f ex5f_1.tmp
runex5f_2:
	-@${MPIEXEC} -n 4 ./ex5f  -da_processors_x 2 -da_processors_y 2 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5f_2.tmp 2>&1; \
	   if (${DIFF} output/ex5f_2.out ex5f_2.tmp) then true; \
	   else echo "Possible problem with ex5f_2, diffs above"; fi; \
	   ${RM} -f ex5f_2.tmp
runex5f_3:
	-@${MPIEXEC} -n 3 ./ex5f -snes_fd  -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5f_3.tmp 2>&1;\
	   if (${DIFF} output/ex5f_3.out ex5f_3.tmp) then true; \
	   else echo "Possible problem with ex5f_3, diffs above"; fi; \
	   ${RM} -f ex5f_3.tmp
runex5f_4:
	-@${MPIEXEC} -n 2 ./ex5f -adifor_jacobian -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5f_4.tmp 2>&1;\
	   if (${DIFF} output/ex5f_4.out ex5f_4.tmp) then true; \
	   else echo "Possible problem with ex5f_4, diffs above"; fi; \
	   ${RM} -f ex5f_4.tmp
runex5f_5:
	-@${MPIEXEC} -n 2 ./ex5f -adiformf_jacobian  -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5f_5.tmp 2>&1;\
	   if (${DIFF} output/ex5f_5.out ex5f_5.tmp) then true; \
	   else echo "Possible problem with ex5f_5, diffs above"; fi; \
	   ${RM} -f ex5f_5.tmp
testex5f: ex5f.PETSc
	@if [ "${PETSC_WITH_BATCH}" != "" ]; then \
           echo "Running with batch filesystem; to test run src/snes/examples/tutorials/ex5f with" ; \
           echo "your systems batch system"; \
        elif [ "${MPIEXEC}" = "/bin/false" ]; then \
           echo "*mpiexec not found*. Please run src/snes/examples/tutorials/ex5f manually"; \
        elif [ -f ex5f ]; then \
	   ${MPIEXEC} -n 1 ./ex5f > ex5f_1.tmp 2>&1; \
	   if (${DIFF} output/ex5f_1.testout ex5f_1.tmp > /dev/null 2>&1) then \
           echo "Fortran example src/snes/examples/tutorials/ex5f run successfully with 1 MPI process"; \
	   else echo "Possible error running Fortran example src/snes/examples/tutorials/ex5f with 1 MPI process"; \
           echo "See http://www.mcs.anl.gov/petsc/petsc-as/documentation/faq.html";\
           cat ex5f_1.tmp; fi;  \
         ${RM} -f ex5f_1.tmp ;\
         ${MAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ex5f.rm; fi
runex5f90:
	-@${MPIEXEC} -n 4 ./ex5f90 -snes_mf -da_processors_x 4 -da_processors_y 1 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5f90_1.tmp 2>&1; \
	   if (${DIFF} output/ex5f90_1.out ex5f90_1.tmp) then true; \
	   else echo "Possible problem with ex5f90_1, diffs above"; fi; \
	   ${RM} -f ex5f90_1.tmp
runex5f90_2:
	-@${MPIEXEC} -n 4 ./ex5f90 -da_processors_x 2 -da_processors_y 2 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5f90_2.tmp 2>&1; \
	   if (${DIFF} output/ex5f90_2.out ex5f90_2.tmp) then true; \
	   else echo "Possible problem with ex5f90_2, diffs above"; fi; \
	   ${RM} -f ex5f90_2.tmp
runex5f90_3:
	-@${MPIEXEC} -n 3 ./ex5f90 -snes_fd  -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5f90_3.tmp 2>&1;\
	   if (${DIFF} output/ex5f90_3.out ex5f90_3.tmp) then true; \
	   else echo "Possible problem with ex5f90_3, diffs above"; fi; \
	   ${RM} -f ex5f90_3.tmp
runex5f90_4:
	-@${MPIEXEC} -n 3 ./ex5f90 -snes_mf_operator  -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5f90_4.tmp 2>&1;\
	   if (${DIFF} output/ex5f90_4.out ex5f90_4.tmp) then true; \
	   else echo "Possible problem with ex5f90_4, diffs above"; fi; \
	   ${RM} -f ex5f90_4.tmp
runex5f90_5:
	-@${MPIEXEC} -n 1 ./ex5f90 -test_appctx > ex5f90_5.tmp 2>&1;\
	   if (${DIFF} output/ex5f90_5.out ex5f90_5.tmp) then true; \
	   else echo "Possible problem with ex5f90_5, diffs above"; fi; \
	   ${RM} -f ex5f90_5.tmp
runex5f90t:
	-@${MPIEXEC} -n 4 ./ex5f90t -snes_mf -da_processors_x 4 -da_processors_y 1 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5f90t_1.tmp 2>&1; \
	   if (${DIFF} output/ex5f90_1.out ex5f90t_1.tmp) then true; \
	   else echo "Possible problem with ex5f90t_1, diffs above"; fi; \
	   ${RM} -f ex5f90t_1.tmp
runex6:
	-@${MPIEXEC} -n 1 ./ex6 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex6_1.tmp 2>&1;   \
	   if (${DIFF} output/ex6_1.out ex6_1.tmp) then true; \
	   else echo "Possible problem with ex6_1, diffs above"; fi; \
	   ${RM} -f ex6_1.tmp
runex6_2:
	-@${MPIEXEC} -n 1 ./ex6 -snes_mf -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex6_2.tmp 2>&1;   \
	   if (${DIFF} output/ex6_2.out ex6_2.tmp) then true; \
	   else echo "Possible problem with ex6_2, diffs above"; fi; \
	   ${RM} -f ex6_2.tmp
runex8:
	-@${MPIEXEC} -n 2 ./ex8  -mx 16 -my 16 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always \
                    -lidvelocity 0. -grashof 1000. > ex8_1.tmp 2>&1; \
           if (${DIFF} output/ex8_1.out ex8_1.tmp) then true; \
           else echo "Possible problem with ex8_1, diffs above"; fi; \
           ${RM} -f ex8_1.tmp
runex8_2:
	-@${MPIEXEC} -n 2 ./ex8 -snes_monitor_solution -ksp_gmres_cgs_refinement_type refine_always > /dev/null 2>&1 
runex8_3:
	-@${MPIEXEC} -n 2 ./ex8 -is_coloring_view > ex8_3.tmp 2>&1;\
           if (${DIFF} output/ex8_3.out ex8_3.tmp) then true; \
           else echo "Possible problem with ex8_3, diffs above"; fi; \
           ${RM} -f ex8_3.tmp
runex14:
	-@${MPIEXEC} -n 4 ./ex14 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex14_1.tmp 2>&1; \
	   if (${DIFF} output/ex14_1.out ex14_1.tmp) then true; \
	   else echo "Possible problem with ex14_1, diffs above"; fi; \
	   ${RM} -f ex14_1.tmp
runex14_2:
	-@${MPIEXEC} -n 4 ./ex14 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always  > ex14_2.tmp 2>&1; \
	   if (${DIFF} output/ex14_2.out ex14_2.tmp) then true; \
	   else echo "Possible problem with ex14_2, diffs above"; fi; \
	   ${RM} -f ex14_2.tmp
runex50:
	-@${MPIEXEC} -n 2 ./ex50 -da_refine 3 -snes_monitor_short -pc_type mg -ksp_type fgmres -pc_mg_type full > ex19_1.tmp 2>&1; \
	   if (${DIFF} output/ex19_1.out ex19_1.tmp) then true; \
	   else echo "Possible problem with ex50, diffs above"; fi; \
	   ${RM} -f ex19_1.tmp
runex50_3: #test pc_redundant
	-@${MPIEXEC} -n 4 ./ex50 -da_refine 3 -snes_monitor_short -pc_type redundant -mat_type mpiaij -redundant_ksp_type preonly -redundant_pc_factor_mat_solver_package mumps -pc_redundant_number 2 > ex19_3.tmp 2>&1; \
	   if (${DIFF} output/ex19_3.out ex19_3.tmp) then true; \
	   else echo "Possible problem with ex50_3, diffs above"; fi; \
	   ${RM} -f ex19_3.tmp
runex50_4: #test pc_redundant
	-@${MPIEXEC} -n 12 ./ex50 -da_refine 3 -snes_monitor_short -pc_type redundant -mat_type mpiaij -redundant_ksp_type preonly -redundant_pc_factor_mat_solver_package mumps -pc_redundant_number 5 > ex19_4.tmp 2>&1; \
	   if (${DIFF} output/ex19_3.out ex19_4.tmp) then true; \
	   else echo "Possible problem with ex50_4, diffs above"; fi; \
	   ${RM} -f ex19_4.tmp
runex50_5: #test different scatters
	-@for A in " " -vecscatter_rsend -vecscatter_ssend -vecscatter_alltoall "-vecscatter_alltoall -vecscatter_nopack" -vecscatter_window; do \
           for B in " " -vecscatter_merge ; do \
             ${MPIEXEC} -n 4 ./ex50 -da_refine 3 -ksp_type fgmres -pc_type mg -pc_mg_type full -snes_monitor_short $$A $$B > ex19_5.tmp 2>&1; \
	     if (${DIFF} output/ex19_5.out ex19_5.tmp) then true; \
	     else echo "Possible problem with ex50_5 " $$A $$B " diffs above"; fi; \
           done;\
         done;\
	 ${RM} -f ex19_5.tmp
runex19:
	-@${MPIEXEC} -n 2 ./ex19 -dmmg_nlevels 4 -snes_monitor_short  > ex19_1.tmp 2>&1; \
	   if (${DIFF} output/ex19_1.out ex19_1.tmp) then true; \
	   else echo "Possible problem with ex19, diffs above"; fi; \
	   ${RM} -f ex19_1.tmp
runex19_2:
	-@${MPIEXEC} -n 2 ./ex19 -dmmg_nlevels 4 -snes_monitor_short -dmmg_jacobian_fd -dmmg_iscoloring_type ghosted > ex19_2.tmp 2>&1; \
	   if (${DIFF} output/ex19_1.out ex19_2.tmp) then true; \
	   else echo "Possible problem with ex19_2, diffs above"; fi; \
	   ${RM} -f ex19_2.tmp
runex19_3: #test pc_redundant
	-@${MPIEXEC} -n 4 ./ex19 -dmmg_nlevels 4 -snes_monitor_short -pc_type redundant -mat_type mpiaij -redundant_ksp_type preonly -redundant_pc_factor_mat_solver_package mumps -pc_redundant_number 2 > ex19_3.tmp 2>&1; \
	   if (${DIFF} output/ex19_3.out ex19_3.tmp) then true; \
	   else echo "Possible problem with ex19_3, diffs above"; fi; \
	   ${RM} -f ex19_3.tmp
runex19_4: #test pc_redundant
	-@${MPIEXEC} -n 12 ./ex19 -dmmg_nlevels 4 -snes_monitor_short -pc_type redundant -mat_type mpiaij -redundant_ksp_type preonly -redundant_pc_factor_mat_solver_package mumps -pc_redundant_number 5 > ex19_4.tmp 2>&1; \
	   if (${DIFF} output/ex19_3.out ex19_4.tmp) then true; \
	   else echo "Possible problem with ex19_4, diffs above"; fi; \
	   ${RM} -f ex19_4.tmp
runex19_5: #test different scatters
	-@for A in " " -vecscatter_rsend -vecscatter_ssend -vecscatter_alltoall "-vecscatter_alltoall -vecscatter_nopack" -vecscatter_window; do \
           for B in " " -vecscatter_merge ; do \
             ${MPIEXEC} -n 4 ./ex19 -dmmg_nlevels 4 -snes_monitor_short $$A $$B > ex19_5.tmp 2>&1; \
	     if (${DIFF} output/ex19_5.out ex19_5.tmp) then true; \
	     else echo "Possible problem with ex19_5 " $$A $$B " diffs above"; fi; \
           done;\
         done;\
	 ${RM} -f ex19_5.tmp

# fieldsplit preconditioner tests
runex19_6: 
	-@${MPIEXEC} -n 1 ./ex19 -snes_monitor_short -ksp_monitor_short -pc_type fieldsplit -snes_view  -preload off > ex19_6.tmp 2>&1; \
	   if (${DIFF} output/ex19_6.out ex19_6.tmp) then true; \
	   else echo "Possible problem with ex19_6, diffs above"; fi; \
	   ${RM} -f ex19_6.tmp

runex19_fieldsplit_2:
	-@${MPIEXEC} -n 1 ./ex19 -dmmg_nlevels 1 -pc_type fieldsplit -pc_fieldsplit_block_size 4 -pc_fieldsplit_type additive -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 -snes_monitor_short -ksp_monitor_short -preload off > ex19_6.tmp 2>&1; \
	   if (${DIFF} output/ex19_fieldsplit_2.out ex19_6.tmp) then true; \
	   else echo "Possible problem with ex19_fieldsplit_2, diffs above"; fi; \
	   ${RM} -f ex19_6.tmp

runex19_fieldsplit_3:
	-@${MPIEXEC} -n 1 ./ex19 -dmmg_nlevels 1 -pc_type fieldsplit -pc_fieldsplit_block_size 4 -pc_fieldsplit_type additive -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 -fieldsplit_0_pc_type lu -fieldsplit_1_pc_type lu -snes_monitor_short -ksp_monitor_short -preload off > ex19_6.tmp 2>&1; \
	   if (${DIFF} output/ex19_fieldsplit_3.out ex19_6.tmp) then true; \
	   else echo "Possible problem with ex19_fieldsplit_3, diffs above"; fi; \
	   ${RM} -f ex19_6.tmp

runex19_fieldsplit_4:
	-@${MPIEXEC} -n 1 ./ex19 -dmmg_nlevels 1 -pc_type fieldsplit -pc_fieldsplit_block_size 4 -pc_fieldsplit_type SCHUR -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 -fieldsplit_0_pc_type lu -fieldsplit_1_pc_type lu -snes_monitor_short -ksp_monitor_short -preload off > ex19_6.tmp 2>&1; \
	   if (${DIFF} output/ex19_fieldsplit_4.out ex19_6.tmp) then true; \
	   else echo "Possible problem with ex19_fieldsplit_4, diffs above"; fi; \
	   ${RM} -f ex19_6.tmp

runex19_fieldsplit_mumps:
	-@${MPIEXEC} -n 2 ./ex19 -dmmg_nlevels 1 -pc_type fieldsplit -pc_fieldsplit_block_size 4 -pc_fieldsplit_type SCHUR -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 -fieldsplit_0_pc_type lu -fieldsplit_1_pc_type lu -snes_monitor_short -ksp_monitor_short -preload off -fieldsplit_0_pc_factor_mat_solver_package mumps -fieldsplit_1_pc_factor_mat_solver_package mumps > ex19_6.tmp 2>&1; \
	   if (${DIFF} output/ex19_fieldsplit_5.out ex19_6.tmp) then true; \
	   else echo "Possible problem with ex19_fieldsplit_5, diffs above"; fi; \
	   ${RM} -f ex19_6.tmp

runex19_fieldsplit_hypre:
	-@${MPIEXEC} -n 2 ./ex19 -dmmg_nlevels 1 -pc_type fieldsplit -pc_fieldsplit_block_size 4 -pc_fieldsplit_type SCHUR -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 -fieldsplit_0_pc_type lu -fieldsplit_0_pc_factor_mat_solver_package mumps -fieldsplit_1_pc_type hypre -fieldsplit_1_pc_hypre_type boomeramg -snes_monitor_short -ksp_monitor_short -preload off > ex19_6.tmp 2>&1; \
	   if (${DIFF} output/ex19_fieldsplit_hypre.out ex19_6.tmp) then true; \
	   else echo "Possible problem with ex19_fieldsplit_hypre, diffs above"; fi; \
	   ${RM} -f ex19_6.tmp

runex19_fieldsplit_redundant:
	-@${MPIEXEC} -n 4 ./ex19 -dmmg_nlevels 1 -pc_type fieldsplit -pc_fieldsplit_block_size 4 -pc_fieldsplit_type SCHUR -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 -fieldsplit_0_pc_type lu -fieldsplit_0_pc_factor_mat_solver_package mumps -fieldsplit_1_pc_type redundant -fieldsplit_1_pc_redundant_number 2 -fieldsplit_1_redundant_ksp_type preonly -fieldsplit_1_redundant_pc_factor_mat_solver_package mumps -snes_monitor_short -ksp_monitor_short -preload off > ex19_6.tmp 2>&1; \
	   if (${DIFF} output/ex19_fieldsplit_redundant.out ex19_6.tmp) then true; \
	   else echo "Possible problem with ex19_fieldsplit_redundant, diffs above"; fi; \
	   ${RM} -f ex19_6.tmp

runex19_composite_fieldsplit: #similar to runex19_fieldsplit_2
	-@${MPIEXEC} -n 1 ./ex19 -dmmg_nlevels 1 -pc_type composite -pc_composite_type MULTIPLICATIVE -pc_composite_pcs fieldsplit,none -sub_0_pc_fieldsplit_block_size 4 -sub_0_pc_fieldsplit_type additive -sub_0_pc_fieldsplit_0_fields 0,1,2 -sub_0_pc_fieldsplit_1_fields 3 -snes_monitor_short -ksp_monitor_short -preload off > ex19_6.tmp 2>&1; \
	   if (${DIFF} output/ex19_composite_fieldsplit.out ex19_6.tmp) then true; \
	   else echo "Possible problem with ex19_composite_fieldsplit, diffs above"; fi; \
	   ${RM} -f ex19_6.tmp

runex19_composite_fieldsplit_bjacobi:
	-@${MPIEXEC} -n 1 ./ex19 -dmmg_nlevels 1 -pc_type composite -pc_composite_type MULTIPLICATIVE -pc_composite_pcs fieldsplit,bjacobi -sub_0_pc_fieldsplit_block_size 4 -sub_0_pc_fieldsplit_type additive -sub_0_pc_fieldsplit_0_fields 0,1,2 -sub_0_pc_fieldsplit_1_fields 3 -sub_1_pc_bjacobi_blocks 16 -sub_1_sub_pc_type lu -snes_monitor_short -ksp_monitor_short -preload off > ex19_6.tmp 2>&1; \
	   if (${DIFF} output/ex19_composite_fieldsplit_bjacobi.out ex19_6.tmp) then true; \
	   else echo "Possible problem with ex19_composite_fieldsplit_bjacobi, diffs above"; fi; \
	   ${RM} -f ex19_6.tmp

runex19_composite_fieldsplit_bjacobi_2:
	-@${MPIEXEC} -n 4 ./ex19 -dmmg_nlevels 1 -pc_type composite -pc_composite_type MULTIPLICATIVE -pc_composite_pcs fieldsplit,bjacobi -sub_0_pc_fieldsplit_block_size 4 -sub_0_pc_fieldsplit_type additive -sub_0_pc_fieldsplit_0_fields 0,1,2 -sub_0_pc_fieldsplit_1_fields 3 -sub_1_pc_bjacobi_blocks 16 -sub_1_sub_pc_type lu -snes_monitor_short -ksp_monitor_short -preload off > ex19_6.tmp 2>&1; \
	   if (${DIFF} output/ex19_composite_fieldsplit_bjacobi_2.out ex19_6.tmp) then true; \
	   else echo "Possible problem with ex19_composite_fieldsplit_bjacobi_2, diffs above"; fi; \
	   ${RM} -f ex19_6.tmp

runex19_7: 
	-@${MPIEXEC} -n 3 ./ex19 -snes_monitor_short -ksp_monitor_short -pc_type fieldsplit -snes_view  -preload off > ex19_7.tmp 2>&1; \
	   if (${DIFF} output/ex19_7.out ex19_7.tmp) then true; \
	   else echo "Possible problem with ex19_7, diffs above"; fi; \
	   ${RM} -f ex19_7.tmp
runex19_8: 
	-@${MPIEXEC} -n 1 ./ex19 -snes_monitor_short -ksp_monitor_short -pc_type fieldsplit -pc_fieldsplit_block_size 2 -pc_fieldsplit_0_fields 0,1 -pc_fieldsplit_1_fields 0,1 -pc_fieldsplit_type multiplicative -snes_view  -preload off -fieldsplit_pc_type lu > ex19_8.tmp 2>&1; \
	   if (${DIFF} output/ex19_8.out ex19_8.tmp) then true; \
	   else echo "Possible problem with ex19_8, diffs above"; fi; \
	   ${RM} -f ex19_8.tmp
runex19_9: 
	-@${MPIEXEC} -n 3 ./ex19 -snes_monitor_short -ksp_monitor_short -pc_type fieldsplit -pc_fieldsplit_type multiplicative -snes_view  -preload off > ex19_9.tmp 2>&1; \
	   if (${DIFF} output/ex19_9.out ex19_9.tmp) then true; \
	   else echo "Possible problem with ex19_9, diffs above"; fi; \
	   ${RM} -f ex19_9.tmp
runex19_10: 
	-@${MPIEXEC} -n 3 ./ex19 -snes_monitor_short -ksp_monitor_short -pc_type fieldsplit -pc_fieldsplit_type symmetric_multiplicative -snes_view  -preload off > ex19_10.tmp 2>&1; \
	   if (${DIFF} output/ex19_10.out ex19_10.tmp) then true; \
	   else echo "Possible problem with ex19_10, diffs above"; fi; \
	   ${RM} -f ex19_10.tmp
runex19_11: #test pc_redundant
	-@${MPIEXEC} -n 4 ./ex19 -dmmg_nlevels 4 -snes_monitor_short -pc_type redundant -mat_type mpiaij -redundant_pc_factor_mat_solver_package pastix -pc_redundant_number 2 -preload off > ex19_11.tmp 2>&1; \
	   if (${DIFF} output/ex19_11.out ex19_11.tmp) then true; \
	   else echo "Possible problem with ex19_11, diffs above"; fi; \
	   ${RM} -f ex19_11.tmp
runex19_12: #test pc_redundant
	-@${MPIEXEC} -n 12 ./ex19 -dmmg_nlevels 4 -snes_monitor_short -pc_type redundant -mat_type mpiaij -redundant_pc_factor_mat_solver_package pastix -pc_redundant_number 5 -preload off > ex19_12.tmp 2>&1; \
	   if (${DIFF} output/ex19_12.out ex19_12.tmp) then true; \
	   else echo "Possible problem with ex19_12, diffs above"; fi; \
	   ${RM} -f ex19_12.tmp
runex19_13: #test pc_fieldsplit with -snes_mf_operator
	-@${MPIEXEC} -n 3 ./ex19 -snes_monitor_short -ksp_monitor_short -pc_type fieldsplit -pc_fieldsplit_type multiplicative -snes_view  -preload off -snes_mf_operator > ex19_13.tmp 2>&1; \
	   if (${DIFF} output/ex19_13.out ex19_13.tmp) then true; \
	   else echo "Possible problem with ex19_13, diffs above"; fi; \
	   ${RM} -f ex19_13.tmp


testex19: ex19.PETSc
	-@if [ "${PETSC_WITH_BATCH}" != "" ]; then \
           echo "Running with batch filesystem; to test run src/snes/examples/tutorials/ex19 with" ; \
           echo "your systems batch system"; \
        elif [ "${MPIEXEC}" = "/bin/false" ]; then \
           echo "*mpiexec not found*. Please run src/snes/examples/tutorials/ex19 manually"; \
	elif [ -f ex19 ]; then \
           ${MPIEXEC} -n 1 ./ex19 -dmmg_nlevels 4  > ex19_1.tmp 2>&1; \
	   if (${DIFF} output/ex19_1.testout ex19_1.tmp > /dev/null 2>&1) then \
           echo "C/C++ example src/snes/examples/tutorials/ex19 run successfully with 1 MPI process"; \
	   else echo "Possible error running C/C++ src/snes/examples/tutorials/ex19 with 1 MPI process"; \
           echo "See http://www.mcs.anl.gov/petsc/petsc-as/documentation/faq.html";\
           cat ex19_1.tmp; fi; \
	if [ "${MPIEXEC}" != "${PETSC_DIR}/bin/mpiexec.uni" ]; then \
           ${MPIEXEC} -n 2 ./ex19 -dmmg_nlevels 4   > ex19_1.tmp 2>&1; \
	   if (${DIFF} output/ex19_1.testout ex19_1.tmp > /dev/null 2>&1) then \
           echo "C/C++ example src/snes/examples/tutorials/ex19 run successfully with 2 MPI processes"; \
	   else echo "Possible error running C/C++ src/snes/examples/tutorials/ex19 with 2 MPI processes"; \
           echo "See http://www.mcs.anl.gov/petsc/petsc-as/documentation/faq.html";\
           cat ex19_1.tmp; fi; fi; \
        ${RM} -f ex19_1.tmp; \
        ${MAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ex19.rm ; fi

testx11ex19: ex19.PETSc
	-@if [ "${PETSC_WITH_BATCH}" != "" ]; then \
           echo "Running with batch filesystem; to test run src/snes/examples/tutorials/ex19 with" ; \
           echo "your systems batch system"; \
        elif [ "${MPIEXEC}" = "/bin/false" ]; then \
           echo "*mpiexec not found*. Please run src/snes/examples/tutorials/ex19 manually"; \
	elif [ -f ex19 ]; then \
          D2=`grep "#define PETSC_HAVE_X11 1" ${PETSC_DIR}/${PETSC_ARCH}/include/petscconf.h`; \
          if [ "$$?" = 0 ]; then \
             ${MPIEXEC} -n 1 ./ex19 -dmmg_nlevels 4 -snes_monitor_draw  > ex19_1.tmp 2>&1; \
	     if (${DIFF} output/ex19_1.testout ex19_1.tmp > /dev/null 2>&1) then \
             echo "Graphics example src/snes/examples/tutorials/ex19 run successfully with 1 MPI process"; \
	     else echo "Possible error running Graphics examples src/snes/examples/tutorials/ex19 1 MPI process"; \
             echo "See http://www.mcs.anl.gov/petsc/petsc-as/documentation/faq.html";\
             cat ex19_1.tmp; fi; \
	  else echo "PETSc installed without X11. Not running src/snes/examples/tutorials/ex19"; fi; \
        ${RM} -f ex19_1.tmp; \
        ${MAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ex19.rm ; fi
runex19_14:
	-@${MPIEXEC} -n 2 ./ex19 -dmmg_nlevels 4 -snes_monitor_short -dmmg_mat_type baij -mg_coarse_pc_type bjacobi > ex19_14.tmp 2>&1; \
	   if (${DIFF} output/ex19_14.out ex19_14.tmp) then true; \
	   else echo "Possible problem with ex19_14, diffs above"; fi; \
	   ${RM} -f ex19_14.tmp
runex19_15:
	-@${MPIEXEC} -n 2 ./ex19 -dmmg_nlevels 4 -snes_monitor_short -dmmg_mat_type baij -mg_coarse_pc_type bjacobi -dmmg_iscoloring_type global > ex19_15.tmp 2>&1; \
	   if (${DIFF} output/ex19_15.out ex19_15.tmp) then true; \
	   else echo "Possible problem with ex19_15, diffs above"; fi; \
	   ${RM} -f ex19_15.tmp
runex19_15a:
	-@${MPIEXEC} -n 2 ./ex19 -dmmg_nlevels 4 -snes_monitor_short -dmmg_mat_type baij -mg_coarse_pc_type bjacobi -dmmg_iscoloring_type global -dmmg_coloring_from_mat > ex19_15a.tmp 2>&1; \
	   if (${DIFF} output/ex19_15a.out ex19_15a.tmp) then true; \
	   else echo "Possible problem with ex19_15a, diffs above"; fi; \
	   ${RM} -f ex19_15a.tmp
runex19_16:
	-@${MPIEXEC} -n 2 ./ex19 -dmmg_nlevels 4 -snes_monitor_short -dmmg_mat_type aij -mg_coarse_pc_type bjacobi -dmmg_iscoloring_type global > ex19_16.tmp 2>&1; \
	   if (${DIFF} output/ex19_16.out ex19_16.tmp) then true; \
	   else echo "Possible problem with ex19_16, diffs above"; fi; \
	   ${RM} -f ex19_16.tmp
runex19_16a:
	-@${MPIEXEC} -n 2 ./ex19 -dmmg_nlevels 4 -snes_monitor_short -dmmg_mat_type aij -mg_coarse_pc_type bjacobi -dmmg_iscoloring_type global -dmmg_coloring_from_mat > ex19_16a.tmp 2>&1; \
	   if (${DIFF} output/ex19_16a.out ex19_16a.tmp) then true; \
	   else echo "Possible problem with ex19_16a, diffs above"; fi; \
	   ${RM} -f ex19_16a.tmp
runex19_superlu:
	-@${MPIEXEC} -n 1 ./ex19 -nlevels 1 -da_grid_x 20 -da_grid_y 20 -pc_type lu -pc_factor_mat_solver_package superlu > ex19.tmp 2>&1; \
	   if (${DIFF} output/ex19_superlu.out ex19.tmp) then true; \
	   else echo "Possible problem with ex19_superlu, diffs above"; fi; \
	   ${RM} -f ex19.tmp
runex19_superlu_equil: #This test fails - I'll check it. Hong
	-@${MPIEXEC} -n 1 ./ex19 -nlevels 1 -da_grid_x 20 -da_grid_y 20 -{snes,ksp}_monitor_short -pc_type lu -pc_factor_mat_solver_package superlu -mat_superlu_equil > ex19.tmp 2>&1; \
	   if (${DIFF} output/ex19_superlu_equil.out ex19.tmp) then true; \
	   else echo "Possible problem with ex19_superlu_equil, diffs above"; fi; \
	   ${RM} -f ex19.tmp
runex19_superlu_dist:
	-@${MPIEXEC} -n 1 ./ex19 -nlevels 1 -da_grid_x 20 -da_grid_y 20 -pc_type lu -pc_factor_mat_solver_package superlu_dist > ex19.tmp 2>&1; \
	   if (${DIFF} output/ex19_superlu.out ex19.tmp) then true; \
	   else echo "Possible problem with ex19_superlu_dist, diffs above"; fi; \
	   ${RM} -f ex19.tmp
runex19_superlu_dist_2:
	-@${MPIEXEC} -n 2 ./ex19 -nlevels 1 -da_grid_x 20 -da_grid_y 20 -pc_type lu -pc_factor_mat_solver_package superlu_dist > ex19.tmp 2>&1; \
	   if (${DIFF} output/ex19_superlu.out ex19.tmp) then true; \
	   else echo "Possible problem with ex19_superlu_dist, diffs above"; fi; \
	   ${RM} -f ex19.tmp
runex19_cusp:
	-@${MPIEXEC} -n 1 ./ex19 -da_vec_type cusp -da_mat_type aijcusp -pc_type none -dmmg_nlevels 1 -snes_monitor_short -snes_rtol 1.e-5 > ex19.tmp 2>&1; \
	   if (${DIFF} output/ex19_cusp.out ex19.tmp) then true; \
	   else echo "Possible problem with ex19_cusp, diffs above"; fi; \
	   ${RM} -f ex19.tmp
runex21:
	-@${MPIEXEC} -n 4 ./ex21  -snes_mf -snes_monitor_short -nox > ex21_1.tmp 2>&1; \
	   if (${DIFF} output/ex21_1.out ex21_1.tmp) then true; \
	   else echo "Possible problem with ex21_1, diffs above"; fi; \
	   ${RM} -f ex21_1.tmp
runex22:
	-@${MPIEXEC} -n 1 ./ex22 -da_grid_x 10 > ex22_1.tmp 2>&1; \
	   if (${DIFF} output/ex22_1.out ex22_1.tmp) then true; \
	   else echo "Possible problem with ex22_1, diffs above"; fi; \
	   ${RM} -f ex22_1.tmp
runex23:
	-@${MPIEXEC} -n 2 ./ex23 -snes_monitor_short -dmmg_jacobian_mf_fd -mat_mffd_type wp -mat_mffd_compute_normu no > ex23_1.tmp 2>&1; \
	   if (${DIFF} output/ex23_1.out ex23_1.tmp) then true; \
	   else echo "Possible problem with ex23_1, diffs above"; fi; \
	   ${RM} -f ex23_1.tmp

runex25:
	-@${MPIEXEC} -n 1 ./ex25 -preload off > ex25_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex25_1.out ex25_1.tmp) then true; \
	   else echo "Possible problem with ex25_1, diffs above"; fi; \
	   ${RM} -f ex25_1.tmp

runex25_2:
	-@${MPIEXEC} -n 2 ./ex25 -preload off > ex25_2.tmp 2>&1;	  \
	   if (${DIFF} output/ex25_2.out ex25_2.tmp) then true; \
	   else echo "Possible problem with ex25_2, diffs above"; fi; \
	   ${RM} -f ex25_2.tmp
runex27:
	-@${MPIEXEC} -n 1 ./ex27 -lidvelocity 100 -da_grid_x 16 -da_grid_y 16 -print -snes_monitor -grashof 1.e5 -cfl_ini 1.0e-1 -snes_rtol 1.0e-3
runex27_p:
	-@${MPIEXEC} -n 1 ./ex27 -lidvelocity 100 -da_grid_x 16 -da_grid_y 16 -print -snes_monitor -grashof 1.e5 -cfl_ini 1.0e-1 -snes_rtol 1.0e-3 -use_parabolic
runex27_li:              # This algorithm, using a linearly implicit step, is the one described in the paper
	-@${MPIEXEC} -n 1 ./ex27 -lidvelocity 100 -da_grid_x 16 -da_grid_y 16 -print -snes_type ksponly -grashof 1.e5 -cfl_ini 1.0e-1 -snes_rtol 1.0e-3

runex28_0:
	-@${MPIEXEC} -n 3 ./ex28 -da_grid_x 20 -snes_converged_reason -snes_monitor_short -problem_type 0 > ex28_0.tmp 2>&1; \
	  ${DIFF} output/ex28_0.out ex28_0.tmp || echo "Possible problem with ex28_0, diffs above"; ${RM} -f ex28_0.tmp
runex28_1:
	-@${MPIEXEC} -n 3 ./ex28 -da_grid_x 20 -snes_converged_reason -snes_monitor_short -problem_type 1 > ex28_1.tmp 2>&1; \
	  ${DIFF} output/ex28_1.out ex28_1.tmp || echo "Possible problem with ex28_1, diffs above"; ${RM} -f ex28_1.tmp
runex28_2:
	-@${MPIEXEC} -n 3 ./ex28 -da_grid_x 20 -snes_converged_reason -snes_monitor_short -problem_type 2 > ex28_2.tmp 2>&1; \
	  ${DIFF} output/ex28_2.out ex28_2.tmp || echo "Possible problem with ex28_2, diffs above"; ${RM} -f ex28_2.tmp
runex28_3:
	-@for mtype in aij nest; do \
	    ${MPIEXEC} -n 3 ./ex28 -da_grid_x 20 -snes_converged_reason -snes_monitor_short -ksp_monitor_short -problem_type 2 -snes_mf_operator \
	    -pack_dm_mat_type $$mtype -pc_type fieldsplit -pc_fieldsplit_type additive -fieldsplit_u_ksp_type gmres -fieldsplit_k_pc_type jacobi > ex28_3.tmp 2>&1; \
	    ${DIFF} output/ex28_3.out ex28_3.tmp || echo "Possible problem with ex28_3 mtype=$${mtype}; diffs above"; ${RM} -f ex28_3.tmp; \
	  done

runex32:
	-@${MPIEXEC} -n 1 ./ex32 -snes_monitor -dmmg_jacobian_mf_fd_operator -da_grid_x 10 -da_grid_y 10
runex32_2:
	-@${MPIEXEC} -n 1 ./ex32 -snes_monitor -dmmg_jacobian_mf_fd -da_grid_x 10 -da_grid_y 10
runex32_3:
	-@${MPIEXEC} -n 3 ./ex32 -snes_monitor -dmmg_jacobian_mf_fd_operator -da_grid_x 10 -da_grid_y 10
runex32_4:
	-@${MPIEXEC} -n 3 ./ex32 -snes_monitor -dmmg_jacobian_mf_fd -da_grid_x 10 -da_grid_y 10

runex42:
	-@${MPIEXEC} -n 1 ./ex42 -snes_monitor_short -snes_max_it 1000 -snes_rtol 1.e-14  > ex42_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex42_1.out ex42_1.tmp) then true; \
	   else echo "Possible problem with ex42_1, diffs above"; fi; \
	   ${RM} -f ex42_1.tmp

runex44:
	-@${MPIEXEC} -n 1 ./ex44 -n 10 -snes_mf -user_precond > ex44.tmp 2>&1; \
	   if (${DIFF} output/ex44.out ex44.tmp) then true; \
	   else echo "Possible problem with ex44, diffs above"; fi; \
	   ${RM} -f ex44.tmp
runex44_2:
	-@${MPIEXEC} -n 1 ./ex44 -n 10 -snes_mf > ex44.tmp 2>&1; \
	   if (${DIFF} output/ex44.out ex44.tmp) then true; \
	   else echo "Possible problem with ex44_2, diffs above"; fi; \
	   ${RM} -f ex44.tmp

runex45:
	-@${MPIEXEC} -n 1 ./ex45 -n 10 > ex45.tmp 2>&1; \
	   if (${DIFF} output/ex44.out ex45.tmp) then true; \
	   else echo "Possible problem with ex45, diffs above"; fi; \
	   ${RM} -f ex45.tmp
runex45_2:
	-@${MPIEXEC} -n 1 ./ex45 -n 10 -snes_jacobian_default -pc_type jacobi > ex45.tmp 2>&1; \
	   if (${DIFF} output/ex44.out ex45.tmp) then true; \
	   else echo "Possible problem with ex45_2, diffs above"; fi; \
	   ${RM} -f ex45.tmp
runex45_3:
	-@${MPIEXEC} -n 1 ./ex45 -n 10 -snes_jacobian_default -fd_jacobian_coloring -my_jacobian_struct > ex45.tmp 2>&1; \
	   if (${DIFF} output/ex44.out ex45.tmp) then true; \
	   else echo "Possible problem with ex45_3, diffs above"; fi; \
	   ${RM} -f ex45.tmp

runex47cu:
	-@${MPIEXEC} -n 1 ./ex47cu -snes_monitor_short -da_vec_type cusp  > ex47cu_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex47cu_1.out ex47cu_1.tmp) then true; \
	   else echo "Possible problem with ex47cu_1, diffs above"; fi; \
	   ${RM} -f ex47cu_1.tmp

runex48:
	-@${MPIEXEC} -n 1 ./ex48 -M 6 -P 4 -thi_nlevels 2 -snes_monitor_short -snes_converged_reason -ksp_monitor_short -ksp_converged_reason -thi_mat_type sbaij > ex48.tmp 2>&1; \
	   ${DIFF} output/ex48_1.out ex48.tmp || echo "Possible problem with ex48_1, diffs above"; \
	   ${RM} -f ex48.tmp
runex48_2:
	-@${MPIEXEC} -n 2 ./ex48 -M 6 -P 4 -thi_hom z -thi_nlevels 2 -snes_monitor_short -snes_converged_reason -ksp_monitor_short -ksp_converged_reason -thi_mat_type sbaij -mg_levels_pc_type asm -mg_levels_pc_asm_blocks 6 -mg_levels_0_pc_type redundant -dmmg_grid_sequence -thi_mat_partitioning_type current > ex48.tmp 2>&1; \
	   ${DIFF} output/ex48_2.out ex48.tmp || echo "Possible problem with ex48_2, diffs above"; \
	   ${RM} -f ex48.tmp
runex48_3:
	-@${MPIEXEC} -n 3 ./ex48 -M 7 -P 4 -thi_hom z -thi_nlevels 2 -snes_monitor_short -snes_converged_reason -ksp_monitor_short -ksp_converged_reason -thi_mat_type baij -mg_levels_pc_type asm -mg_levels_pc_asm_blocks 9 -thi_mat_partitioning_type current > ex48.tmp 2>&1; \
	   ${DIFF} output/ex48_3.out ex48.tmp || echo "Possible problem with ex48_3, diffs above"; \
	   ${RM} -f ex48.tmp
runex48_4:
	-@${MPIEXEC} -n 6 ./ex48 -M 4 -P 2 -da_refine_hierarchy_x 1,1,3 -da_refine_hierarchy_y 2,2,1 -da_refine_hierarchy_z 2,2,1 -dmmg_grid_sequence 1 -dmmg_view -ksp_converged_reason -ksp_monitor_short -ksp_rtol 1e-2 -mg_levels_1_sub_pc_type cholesky -pc_mg_type multiplicative -snes_converged_reason -snes_monitor_short -snes_stol 1e-12 -thi_L 80e3 -thi_alpha 0.05 -thi_friction_m 1 -thi_hom x -thi_nlevels 4 > ex48.tmp 2>&1; \
	   ${DIFF} output/ex48_4.out ex48.tmp || echo "Possible problem with ex48_4, diffs above"; \
	   ${RM} -f ex48.tmp

testex50: ex50.PETSc
	-@if [ "${PETSC_WITH_BATCH}" != "" ]; then \
           echo "Running with batch filesystem; to test run src/snes/examples/tutorials/ex50 with" ; \
           echo "your systems batch system"; \
        elif [ "${MPIEXEC}" = "/bin/false" ]; then \
           echo "*mpiexec not found*. Please run src/snes/examples/tutorials/ex50 manually"; \
	elif [ -f ex50 ]; then \
           ${MPIEXEC} -n 1 ./ex50 -da_refine 3  > ex50_1.tmp 2>&1; \
	   if (${DIFF} output/ex19_1.testout ex50_1.tmp > /dev/null 2>&1) then \
           echo "C/C++ example src/snes/examples/tutorials/ex50 run successfully with 1 MPI process"; \
	   else echo "Possible error running C/C++ src/snes/examples/tutorials/ex50 with 1 MPI process"; \
           echo "See http://www.mcs.anl.gov/petsc/petsc-as/documentation/faq.html";\
           cat ex50_1.tmp; fi; \
	if [ "${MPIEXEC}" != "${PETSC_DIR}/bin/mpiexec.uni" ]; then \
           ${MPIEXEC} -n 2 ./ex50 -da_refine 3   > ex50_1.tmp 2>&1; \
	   if (${DIFF} output/ex19_1.testout ex50_1.tmp > /dev/null 2>&1) then \
           echo "C/C++ example src/snes/examples/tutorials/ex50 run successfully with 2 MPI processes"; \
	   else echo "Possible error running C/C++ src/snes/examples/tutorials/ex50 with 2 MPI processes"; \
           echo "See http://www.mcs.anl.gov/petsc/petsc-as/documentation/faq.html";\
           cat ex50_1.tmp; fi; fi; \
        ${RM} -f ex50_1.tmp; \
        ${MAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ex50.rm ; fi

testx11ex50: ex50.PETSc
	-@if [ "${PETSC_WITH_BATCH}" != "" ]; then \
           echo "Running with batch filesystem; to test run src/snes/examples/tutorials/ex50 with" ; \
           echo "your systems batch system"; \
        elif [ "${MPIEXEC}" = "/bin/false" ]; then \
           echo "*mpiexec not found*. Please run src/snes/examples/tutorials/ex50 manually"; \
	elif [ -f ex50 ]; then \
          D2=`grep "#define PETSC_HAVE_X11 1" ${PETSC_DIR}/${PETSC_ARCH}/include/petscconf.h`; \
          if [ "$$?" = 0 ]; then \
             ${MPIEXEC} -n 1 ./ex50 -da_refine 3 -snes_monitor_draw  > ex50_1.tmp 2>&1; \
	     if (${DIFF} output/ex19_1.testout ex50_1.tmp > /dev/null 2>&1) then \
             echo "Graphics example src/snes/examples/tutorials/ex50 run successfully with 1 MPI process"; \
	     else echo "Possible error running Graphics examples src/snes/examples/tutorials/ex50 1 MPI process"; \
             echo "See http://www.mcs.anl.gov/petsc/petsc-as/documentation/faq.html";\
             cat ex50_1.tmp; fi; \
	  else echo "PETSc installed without X11. Not running src/snes/examples/tutorials/ex50"; fi; \
        ${RM} -f ex50_1.tmp; \
        ${MAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ex50.rm ; fi

TESTEXAMPLES_C		       = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2  runex2_3 ex2.rm ex3.PETSc runex3 \
                                 runex3_2 runex3_3 runex3_4 ex3.rm ex5.PETSc runex5 runex5_2 runex5_3 runex5_4 runex5_6 ex5.rm\
                                 ex6.PETSc runex6 runex6_2 ex6.rm\
                                 ex14.PETSc runex14 runex14_2 ex14.rm ex19.PETSc runex19 runex19_2 runex19_5 \
                                 runex19_6 runex19_fieldsplit_2 runex19_fieldsplit_3 runex19_fieldsplit_4 \
                                 runex19_composite_fieldsplit runex19_composite_fieldsplit_bjacobi runex19_composite_fieldsplit_bjacobi_2\
                                 runex19_7 runex19_8 runex19_9 \
                                 runex19_10 runex19_14 runex19_15 runex19_15a runex19_16 runex19_16a ex19.rm \
                                 ex22.PETSc runex22 ex22.rm ex18.PETSc ex18.rm ex19.PETSc ex19.rm ex21.PETSc runex21 ex21.rm \
                                 ex23.PETSc runex23 ex23.rm ex24.PETSc ex24.rm   ex25.PETSc runex25 runex25_2 \
                                 ex25.rm ex28.PETSc runex28_0 runex28_1 runex28_2 runex28_3 ex28.rm ex29.PETSc ex29.rm \
                                 ex31.PETSc ex31.rm ex32.PETSc ex32.rm ex42.PETSc runex42 ex42.rm ex43.PETSc ex43.rm \
                                 ex44.PETSc runex44 runex44_2 ex44.rm ex45.PETSc runex45 runex45_2 runex45_3 ex45.rm \
                                 ex48.PETSc runex48 runex48_2 runex48_3 runex48_4 ex48.rm \
                                 ex54.PETSc ex54.rm ex59.PETSc ex59.rm
TESTEXAMPLES_C_X11	       = 
TESTEXAMPLES_FORTRAN	       = ex1f.PETSc runex1f ex1f.rm
TESTEXAMPLES_C_NOCOMPLEX       = ex26.PETSc ex26.rm ex30.PETSc ex30.rm
TESTEXAMPLES_FORTRAN_NOCOMPLEX = ex5f.PETSc runex5f runex5f_3 ex5f.rm
TESTEXAMPLES_FORTRAN_MPIUNI    = ex1f.PETSc runex1f ex1f.rm
TESTEXAMPLES_C_X11_MPIUNI      = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex3.PETSc runex3 ex3.rm ex6.PETSc \
                                 runex6 runex6_2 ex6.rm  ex14.PETSc ex14.rm 
TESTEXAMPLES_F90	       = ex5f90.PETSc runex5f90 runex5f90_2 runex5f90_3 runex5f90_4 runex5f90_5 ex5f90.rm
TESTEXAMPLES_13		       = 
TESTEXAMPLES_MATLAB_ENGINE     = 
TESTEXAMPLES_AMS	       = 
TESTEXAMPLES_ADIFOR	       = ex5f.PETSc runex5f_2 ex5f.rm
TESTEXAMPLES_MUMPS             = ex19.PETSc runex19_3 runex19_4 runex19_fieldsplit_mumps runex19_fieldsplit_redundant ex19.rm  \
                                 ex50.PETSc runex50_3 runex50_4 runex50_fieldsplit_mumps runex50_fieldsplit_redundant ex50.rm 
TESTEXAMPLES_SUPERLU           = ex19.PETSc runex19_superlu ex19.rm
TESTEXAMPLES_SUPERLU_DIST      = ex19.PETSc runex19_superlu_dist runex19_superlu_dist_2 ex19.rm  ex50.PETSc  ex50.rm
TESTEXAMPLES_PASTIX            = ex19.PETSc runex19_11 runex19_12 ex19.rm  ex50.PETSc runex50_11 runex50_12 ex50.rm
TESTEXAMPLES_CUSP              = ex19.PETSc runex19_cusp ex19.rm ex47cu.PETSc runex47cu ex47cu.rm 

include ${PETSC_DIR}/conf/test
