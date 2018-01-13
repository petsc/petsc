CFLAGS	         =
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR           = src/ksp/ksp/examples/tutorials/
DIRS             = network
EXAMPLESC        = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c ex9.c \
                ex10.c ex11.c ex12.c ex13.c ex15.c ex16.c ex18.c ex23.c \
                ex25.c ex27.c ex28.c ex29.c ex30.c ex31.c ex32.c ex34.c \
                ex41.c ex42.c ex43.c \
                ex45.c ex46.c  ex49.c ex50.c ex51.c ex52.c ex53.c \
                ex54.c ex55.c ex56.c ex58.c ex62.c ex63.cxx ex64.c ex65.c ex66.c ex67.c ex68.c ex69.c
EXAMPLESF        = ex1f.F90 ex2f.F ex6f.F ex11f.F ex13f90.F90 ex14f.F ex15f.F ex21f.F ex22f.F ex44f.F90 ex45f.F \
                   ex52f.F ex54f.F90 ex61f.F90
MANSEC           = KSP
CLEANFILES       = rhs.vtk solution.vtk
NP               = 1

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_KSP_LIB}
	${RM} ex1.o

ex1f: ex1f.o  chkopts
	-${FLINKER} -o ex1f ex1f.o  ${PETSC_KSP_LIB}
	${RM} ex1f.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_KSP_LIB}
	${RM} ex2.o

ex2a: ex2a.o  chkopts
	-${CLINKER} -o ex2a ex2a.o  ${PETSC_KSP_LIB}
	${RM} ex2a.o

ex2f: ex2f.o  chkopts
	-${FLINKER} -o ex2f ex2f.o  ${PETSC_KSP_LIB}
	${RM} ex2f.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_KSP_LIB}
	${RM} ex3.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_KSP_LIB}
	${RM} ex4.o

ex5: ex5.o  chkopts
	-${CLINKER} -o ex5 ex5.o  ${PETSC_KSP_LIB}
	${RM} ex5.o

ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_KSP_LIB}
	${RM} ex6.o

ex6f: ex6f.o  chkopts
	-${FLINKER} -o ex6f ex6f.o  ${PETSC_KSP_LIB}
	${RM} ex6f.o

ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_KSP_LIB}
	${RM} ex7.o

ex8: ex8.o  chkopts
	-${CLINKER} -o ex8 ex8.o  ${PETSC_KSP_LIB}
	${RM} ex8.o

ex9: ex9.o  chkopts
	-${CLINKER} -o ex9 ex9.o  ${PETSC_KSP_LIB}
	${RM} ex9.o

ex10: ex10.o  chkopts
	-${CLINKER} -o ex10 ex10.o  ${PETSC_KSP_LIB}
	${RM} ex10.o

ex11: ex11.o  chkopts
	-${CLINKER} -o ex11 ex11.o  ${PETSC_KSP_LIB}
	${RM} ex11.o

ex11f: ex11f.o  chkopts
	-${FLINKER} -o ex11f ex11f.o  ${PETSC_KSP_LIB}
	${RM} ex11f.o

ex12: ex12.o  chkopts
	-${CLINKER} -o ex12 ex12.o  ${PETSC_KSP_LIB}
	${RM} ex12.o

ex13: ex13.o  chkopts
	-${CLINKER} -o ex13 ex13.o  ${PETSC_KSP_LIB}
	${RM} ex13.o

ex13f90: ex13f90.o  chkopts
	-${FLINKER} -o ex13f90 ex13f90.o  ${PETSC_KSP_LIB}
	${RM} ex13f90.o

ex14: ex14.o  chkopts
	-${CLINKER} -o ex14 ex14.o  ${PETSC_KSP_LIB}
	${RM} ex14.o

ex14f: ex14f.o  chkopts
	-${FLINKER} -o ex14f ex14f.o  ${PETSC_KSP_LIB}
	${RM} ex14f.o

ex15: ex15.o  chkopts
	-${CLINKER} -o ex15 ex15.o  ${PETSC_KSP_LIB}
	${RM} ex15.o

ex15f: ex15f.o  chkopts
	-${FLINKER} -o ex15f ex15f.o  ${PETSC_KSP_LIB}
	${RM} ex15f.o

ex16: ex16.o  chkopts
	-${CLINKER} -o ex16 ex16.o  ${PETSC_KSP_LIB}
	${RM} ex16.o

ex17: ex17.o  chkopts
	-${CLINKER} -o ex17 ex17.o  ${PETSC_KSP_LIB}
	${RM} ex17.o

ex18: ex18.o  chkopts
	-${CLINKER} -o ex18 ex18.o  ${PETSC_KSP_LIB}
	${RM} ex18.o

ex20: ex20.o  chkopts
	-${CLINKER} -o ex20 ex20.o  ${PETSC_KSP_LIB}
	${RM} ex20.o

ex21f: ex21f.o  chkopts
	-${FLINKER} -o ex21f ex21f.o  ${PETSC_KSP_LIB}
	${RM} ex21f.o

ex22: ex22.o  chkopts
	-${CLINKER} -o ex22 ex22.o  ${PETSC_SNES_LIB}
	${RM} ex22.o

ex22f: ex22f.o  chkopts
	-${FLINKER} -o ex22f ex22f.o  ${PETSC_SNES_LIB}
	${RM} ex22f.o

ex23: ex23.o  chkopts
	-${CLINKER} -o ex23 ex23.o  ${PETSC_KSP_LIB}
	${RM} ex23.o

ex25: ex25.o  chkopts
	-${CLINKER} -o ex25 ex25.o  ${PETSC_SNES_LIB}
	${RM} ex25.o

ex26: ex26.o  chkopts
	-${CLINKER} -o ex26 ex26.o  ${PETSC_KSP_LIB}
	${RM} ex26.o

ex27: ex27.o  chkopts
	-${CLINKER} -o ex27 ex27.o  ${PETSC_KSP_LIB}
	${RM} ex27.o

ex28: ex28.o  chkopts
	-${CLINKER} -o ex28 ex28.o  ${PETSC_SNES_LIB}
	${RM} ex28.o

ex29: ex29.o  chkopts
	-${CLINKER} -o ex29 ex29.o  ${PETSC_SNES_LIB}
	${RM} ex29.o

ex30: ex30.o  chkopts
	-${CLINKER} -o ex30 ex30.o  ${PETSC_KSP_LIB}
	${RM} ex30.o

ex31: ex31.o  chkopts
	-${CLINKER} -o ex31 ex31.o  ${PETSC_SNES_LIB}
	${RM} ex31.o

ex32: ex32.o  chkopts
	-${CLINKER} -o ex32 ex32.o  ${PETSC_SNES_LIB}
	${RM} ex32.o

ex33: ex33.o chkopts
	-${CLINKER} -o ex33 ex33.o ${PETSC_SNES_LIB}
	${RM} ex33.o

ex34: ex34.o chkopts
	-${CLINKER} -o ex34 ex34.o ${PETSC_SNES_LIB}
	${RM} ex34.o

ex35: ex35.o chkopts
	-${CLINKER} -o ex35 ex35.o ${PETSC_SNES_LIB}
	${RM} ex35.o

ex36: ex36.o chkopts
	-${CLINKER} -o ex36 ex36.o ${PETSC_SNES_LIB}
	${RM} ex36.o

ex37: ex37.o chkopts
	-${CLINKER} -o ex37 ex37.o ${PETSC_SNES_LIB}
	${RM} ex37.o

ex38: ex38.o chkopts
	-${CLINKER} -o ex38 ex38.o ${PETSC_SNES_LIB}
	${RM} ex38.o

ex39: ex39.o chkopts
	-${CLINKER} -o ex39 ex39.o ${PETSC_SNES_LIB}
	${RM} ex39.o

ex40: ex40.o chkopts
	-${CLINKER} -o ex40 ex40.o ${PETSC_SNES_LIB}
	${RM} ex40.o

ex41: ex41.o chkopts
	-${CLINKER} -o ex41 ex41.o ${PETSC_SNES_LIB}
	${RM} ex41.o

ex42: ex42.o chkopts
	-${CLINKER} -o ex42 ex42.o ${PETSC_KSP_LIB}
	${RM} ex42.o

ex43: ex43.o chkopts
	-${CLINKER} -o ex43 ex43.o ${PETSC_KSP_LIB}
	${RM} ex43.o

ex44f: ex44f.o  chkopts
	-${FLINKER} -o ex44f ex44f.o  ${PETSC_KSP_LIB}
	${RM} ex44f.o

ex45f: ex45f.o  chkopts
	-${FLINKER} -o ex45f ex45f.o  ${PETSC_KSP_LIB}
	${RM} ex45f.o

ex45: ex45.o chkopts
	-${CLINKER} -o ex45 ex45.o ${PETSC_KSP_LIB}
	${RM} ex45.o

ex46: ex46.o chkopts
	-${CLINKER} -o ex46 ex46.o ${PETSC_KSP_LIB}
	${RM} ex46.o

ex47: ex47.o chkopts
	-${CLINKER} -o ex47 ex47.o ${PETSC_KSP_LIB}
	${RM} ex47.o

ex49: ex49.o chkopts
	-${CLINKER} -o ex49 ex49.o ${PETSC_KSP_LIB}
	${RM} ex49.o

ex50: ex50.o chkopts
	-${CLINKER} -o ex50 ex50.o ${PETSC_KSP_LIB}
	${RM} ex50.o

ex51: ex51.o chkopts
	-${CLINKER} -o ex51 ex51.o ${PETSC_KSP_LIB}
	${RM} ex51.o

ex52: ex52.o chkopts
	-${CLINKER} -o ex52 ex52.o ${PETSC_KSP_LIB}
	${RM} ex52.o

ex52f: ex52f.o  chkopts
	-${FLINKER} -o ex52f ex52f.o  ${PETSC_KSP_LIB}
	${RM} ex52f.o

ex53: ex53.o chkopts
	-${CLINKER} -o ex53 ex53.o ${PETSC_KSP_LIB}
	${RM} ex53.o

ex54: ex54.o chkopts
	-${CLINKER} -o ex54 ex54.o ${PETSC_KSP_LIB}
	${RM} ex54.o

ex54f: ex54f.o  chkopts
	-${FLINKER} -o ex54f ex54f.o  ${PETSC_KSP_LIB}
	${RM} ex54f.o

ex55: ex55.o chkopts
	-${CLINKER} -o ex55 ex55.o ${PETSC_KSP_LIB}
	${RM} ex55.o

ex56: ex56.o chkopts
	-${CLINKER} -o ex56 ex56.o ${PETSC_KSP_LIB}
	${RM} ex56.o

ex57f: ex57f.o  chkopts
	-${FLINKER} -o ex57f ex57f.o  ${PETSC_KSP_LIB}
	${RM} ex57f.o

ex58: ex58.o chkopts
	-${CLINKER} -o ex58 ex58.o ${PETSC_KSP_LIB}
	${RM} ex58.o

ex59: ex59.o chkopts
	-${CLINKER} -o ex59 ex59.o ${PETSC_KSP_LIB}
	${RM} ex59.o

ex60: ex60.o  chkopts
	-${CLINKER} -o ex60 ex60.o  ${PETSC_KSP_LIB}
	${RM} ex60.o

ex61f: ex61f.o  chkopts
	-${FLINKER} -o ex61f ex61f.o  ${PETSC_KSP_LIB}
	${RM} ex61f.o

ex62: ex62.o  chkopts
	-${CLINKER} -o ex62 ex62.o  ${PETSC_KSP_LIB}
	${RM} ex62.o

ex63: ex63.o  chkopts
	-${CLINKER} -o ex63 ex63.o  ${PETSC_KSP_LIB}
	${RM} ex63.o

ex64: ex64.o  chkopts
	-${CLINKER} -o ex64 ex64.o  ${PETSC_KSP_LIB}
	${RM} ex64.

ex65: ex65.o  chkopts
	-${CLINKER} -o ex65 ex65.o  ${PETSC_KSP_LIB}
	${RM} ex65.o

ex66: ex66.o  chkopts
	-${CLINKER} -o ex66 ex66.o  ${PETSC_KSP_LIB}
	${RM} ex66.o

ex67: ex67.o  chkopts
	-${CLINKER} -o ex67 ex67.o  ${PETSC_KSP_LIB}
	${RM} ex67.o

ex68: ex68.o  chkopts
	-${CLINKER} -o ex68 ex68.o  ${PETSC_KSP_LIB}
	${RM} ex68.o

ex69: ex69.o  chkopts
	-${CLINKER} -o ex69 ex69.o  ${PETSC_KSP_LIB}
	${RM} ex69.o

ex70: ex70.o  chkopts
	-${CLINKER} -o ex70 ex70.o  ${PETSC_KSP_LIB}
	${RM} ex70.o

#----------------------------------------------------------------------------
NP = 1
M  = 4
N  = 5
MDOMAINS = 2
NDOMAINS = 1
OVERLAP=1

runex11:
	-@${MPIEXEC} -n 1 ./ex11 -n 6 -norandom -pc_type none -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex11_1.tmp 2>&1; \
	   if (${DIFF} output/ex11_1.out ex11_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex11_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex11_1.tmp
runex11f:
	-@${MPIEXEC} -n 1 ./ex11f -n 6 -norandom -pc_type none -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex11f_1.tmp 2>&1; \
	   if (${DIFF} output/ex11f_1.out ex11f_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex11f_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex11f_1.tmp
runex12:
	-@${MPIEXEC} -n 1 ./ex12 -ksp_gmres_cgs_refinement_type refine_always > ex12_1.tmp 2>&1; \
	   if (${DIFF} output/ex12_1.out ex12_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex12_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex12_1.tmp
runex13:
	-@${MPIEXEC} -n 1 ./ex13 -m 19 -n 20 -ksp_gmres_cgs_refinement_type refine_always > ex13_1.tmp 2>&1; \
	   if (${DIFF} output/ex13_1.out ex13_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex13_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex13_1.tmp
runex13f90:
	-@${MPIEXEC} -n 1 ./ex13f90 -m 19 -n 20 -ksp_gmres_cgs_refinement_type refine_always > ex13f90_1.tmp 2>&1; \
	   if (${DIFF} output/ex13f90_1.out ex13f90_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex13f90_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex13f90_1.tmp usermodule.mod
runex14f:
	-@${MPIEXEC} -n 1 ./ex14f -no_output -ksp_gmres_cgs_refinement_type refine_always > ex14_1.tmp 2>&1; \
	   if (${DIFF} output/ex14_1.out ex14_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex14f_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex14_1.tmp
runex15:
	-@${MPIEXEC} -n 2 ./ex15 -ksp_view -user_defined_pc -ksp_gmres_cgs_refinement_type refine_always > ex15_1.tmp 2>&1; \
	   if (${DIFF} output/ex15_1.out ex15_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex15_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex15_1.tmp
runex15_tsirm:
	-@${MPIEXEC} -n 1 ./ex15 -m 600 -n 600 -ksp_type tsirm -pc_type ksp -ksp_monitor_short -ksp_ksp_type fgmres -ksp_ksp_rtol 1e-10  -ksp_pc_type mg -ksp_ksp_max_it 30  > ex15_tsirm.tmp 2>&1; \
	   if (${DIFF} output/ex15_tsirm.out ex15_tsirm.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex15_tsirm, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex15_tsirm.tmp
runex15f:
	-@${MPIEXEC} -n 2 ./ex15f -ksp_view -user_defined_pc -ksp_gmres_cgs_refinement_type refine_always > ex15f_1.tmp 2>&1; \
	   if (${DIFF} output/ex15f_1.out ex15f_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex15f_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex15f_1.tmp
runex16:
	-@${MPIEXEC} -n 2 ./ex16 -ntimes 4 -ksp_gmres_cgs_refinement_type refine_always > ex16_1.tmp 2>&1; \
	   if (${DIFF} output/ex16_1.out ex16_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex16_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex16_1.tmp
runex18:
	-@${MPIEXEC} -n 3 ./ex18 -m 39 -n 18 -ksp_monitor_short -permute nd > ex18_1.tmp 2>&1; \
	   ${DIFF} output/ex18_1.out ex18_1.tmp || printf "${PWD}\nPossible problem with ex18_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex18_1.tmp
runex18_bas:
	-@${MPIEXEC} -n 1 ./ex18 -m 13 -n 17 -ksp_monitor_short -ksp_type cg -pc_type icc -pc_factor_mat_solver_package bas -ksp_view -pc_factor_levels 1 | grep -v "variant " > ex18_bas.tmp 2>&1; \
	   ${DIFF} output/ex18_bas.out ex18_bas.tmp || printf "${PWD}\nPossible problem with ex18_bas, diffs above\n=========================================\n"; \
	   ${RM} -f ex18_bas.tmp
runex18_2:
	-@${MPIEXEC} -n 3 ./ex18 -m 39 -n 18 -ksp_monitor_short -permute rcm > ex18_2.tmp 2>&1; \
	   ${DIFF} output/ex18_2.out ex18_2.tmp || printf "${PWD}\nPossible problem with ex18_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex18_2.tmp
runex18_3:
	-@${MPIEXEC} -n 3 ./ex18 -m 13 -n 17 -ksp_monitor_short -ksp_type cg -ksp_cg_single_reduction > ex18_3.tmp 2>&1; \
	   ${DIFF} output/ex18_3.out ex18_3.tmp || printf "${PWD}\nPossible problem with ex18_3, diffs above\n=========================================\n"; \
	   ${RM} -f ex18_3.tmp
runex21f:
	-@${MPIEXEC} -n 1 ./ex21f  > ex21f_1.tmp 2>&1; \
	   if (${DIFF} output/ex21f_1.out ex21f_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex21f_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex21f_1.tmp
runex22f:
	-@${MPIEXEC} -n 1 ./ex22f -pc_mg_type full -ksp_monitor_short -mg_levels_ksp_monitor_short -mg_levels_ksp_norm_type preconditioned -pc_type mg -da_refine 2 -ksp_type fgmres > ex22_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex22_1.out ex22_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex22f_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex22_1.tmp

runex23:
	-@${MPIEXEC} -n 1 ./ex23 -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex23_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex23_1.out ex23_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex23_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex23_1.tmp

runex23_2:
	-@${MPIEXEC} -n 3 ./ex23 -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex23_2.tmp 2>&1;	  \
	   if (${DIFF} output/ex23_2.out ex23_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex23_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex23_2.tmp

runex23_3 :
	-@${MPIEXEC} -n 2 ./ex23 -ksp_monitor_short -ksp_rtol 1e-6 -ksp_type pipefgmres > ex23_3.tmp 2>&1;	  \
	   if (${DIFF} output/ex23_3.out ex23_3.tmp) then true; \
	   else printf "${PWD}\nPossible problem with with ex23_3, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex23_3.tmp

runex25:
	-@${MPIEXEC} -n 1 ./ex25 -pc_type mg -ksp_type fgmres -da_refine 2 -ksp_monitor_short -mg_levels_ksp_monitor_short -mg_levels_ksp_norm_type unpreconditioned -ksp_view -pc_mg_type full  > ex25_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex25_1.out ex25_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex25_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex25_1.tmp

runex25_2:
	-@${MPIEXEC} -n 2 ./ex25 -pc_type mg -ksp_type fgmres -da_refine 2 -ksp_monitor_short -mg_levels_ksp_monitor_short -mg_levels_ksp_norm_type unpreconditioned -ksp_view -pc_mg_type full > ex25_2.tmp 2>&1;	  \
	   if (${DIFF} output/ex25_2.out ex25_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex25_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex25_2.tmp

runex27:
	-@${MPIEXEC} -n 1 ./ex27 -f ${DATAFILESPATH}/matrices/medium  -ksp_view  -ksp_monitor_short -ksp_max_it 100  > ex27_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex27_1.out ex27_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex27_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex27_1.tmp

runex28:
	-@${MPIEXEC} -n 1 ./ex28 -ksp_monitor_short -pc_type mg -pc_mg_type full -ksp_type fgmres -da_refine 2 -mg_levels_ksp_type gmres -mg_levels_ksp_max_it 1 -mg_levels_pc_type ilu > ex28_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex28_1.out ex28_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex28_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex28_1.tmp binaryoutput binaryoutput.info

runex29:
	-@${MPIEXEC} -n 1 ./ex29 -pc_type mg -pc_mg_type full -ksp_type fgmres -ksp_monitor_short -da_refine 8 > ex29_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex29_1.out ex29_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex29_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex29_1.tmp

runex29_2:
	-@${MPIEXEC} -n 1 ./ex29  -bc_type neumann -pc_type mg -pc_mg_type full -ksp_type fgmres -ksp_monitor_short -da_refine 8 -mg_coarse_pc_factor_shift_type nonzero > ex29_2.tmp 2>&1;	  \
	   if (${DIFF} output/ex29_2.out ex29_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex29_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex29_2.tmp

runex29_telescope:
	-@${MPIEXEC} -n 4 ./ex29 -ksp_monitor_short -da_grid_x 257 -da_grid_y 257 -pc_type mg -pc_mg_galerkin pmat -pc_mg_levels 4 -ksp_type richardson -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi -mg_coarse_pc_type telescope -mg_coarse_pc_telescope_ignore_kspcomputeoperators -mg_coarse_telescope_pc_type mg -mg_coarse_telescope_pc_mg_galerkin pmat -mg_coarse_telescope_pc_mg_levels 3 -mg_coarse_telescope_mg_levels_ksp_type chebyshev -mg_coarse_telescope_mg_levels_pc_type jacobi -mg_coarse_pc_telescope_reduction_factor 4 > ex29_telescope.tmp 2>&1;	  \
	   if (${DIFF} output/ex29_telescope.out ex29_telescope.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex29_telescope, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex29_telescope.tmp

runex30:
	-@${MPIEXEC} -n 1 ./ex30 > ex30_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex30_1.out ex30_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex30_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex30_1.tmp

runex32:
	-@${MPIEXEC} -n 1 ./ex32 -pc_type mg -pc_mg_type full -ksp_type fgmres -ksp_monitor_short -pc_mg_levels 3 -mg_coarse_pc_factor_shift_type nonzero > ex32_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex32_1.out ex32_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex32_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex32_1.tmp

runex34:
	-@${MPIEXEC} -n 1 ./ex34  -pc_type mg -pc_mg_type full -ksp_type fgmres -ksp_monitor_short -pc_mg_levels 3 -mg_coarse_pc_factor_shift_type nonzero -ksp_view > ex34_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex34_1.out ex34_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex34_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex34_1.tmp

runex34_hyprestruct:
	-@${MPIEXEC} -n 3 ./ex34 -ksp_type gmres -pc_type pfmg -dm_mat_type hyprestruct  -ksp_monitor -da_refine 3   > ex34_hyprestruct.tmp 2>&1;	  \
	   if (${DIFF} output/ex34_hyprestruct.out ex34_hyprestruct.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex34_hyprestruct, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex34_hyprestruct.tmp

runex34_2 :
	-@${MPIEXEC} -n 2 ./ex34 -ksp_monitor_short -da_grid_x 50 -da_grid_y 50 -pc_type ksp -ksp_ksp_type cg -ksp_pc_type bjacobi -ksp_ksp_rtol 1e-1 -ksp_ksp_monitor -ksp_type pipefgmres -ksp_gmres_restart 5 > ex34_2.tmp 2>&1; \
	   if (${DIFF} output/ex34_2.out ex34_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with with ex34_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex34_2.tmp

runex35:
	-@${MPIEXEC} -n 1 ./ex35 -levels 0 -nu .01 -n 10 -ksp_type cg -pc_type sor -ksp_converged_reason > ex35_1.tmp 2>&1;\
	   if (${DIFF} output/ex35_1.out ex35_1.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with runex35, diffs above \n========================================="; fi ;\
	   ${RM} -f ex35_1.tmp

runex35_2:
	-@${MPIEXEC} -n 2 ./ex35 -levels 3 -nu .01 -n 2 -mg -ksp_converged_reason > ex35_2.tmp 2>&1;\
	   if (${DIFF} output/ex35_2.out ex35_2.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with runex35_2, diffs above \n========================================="; fi ;\
	   ${RM} -f ex35_2.tmp

runex35_3:
	-@${MPIEXEC} -n 2 ./ex35 -problem 3 -file data/ex35_mesh.h5m -mg -levels 1 -ksp_converged_reason > ex35_3.tmp 2>&1;\
	   if (${DIFF} output/ex35_3.out ex35_3.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with runex35_3, diffs above \n========================================="; fi ;\
	   ${RM} -f ex35_3.tmp

runex36:
	-@${MPIEXEC} -n 1 ./ex36 -levels 1 -nu .01 -n 4 -mg -ksp_converged_reason > ex36_1.tmp 2>&1;\
	   if (${DIFF} output/ex36_1.out ex36_1.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with runex36, diffs above \n========================================="; fi ;\
	   ${RM} -f ex36_1.tmp

runex36_2:
	-@${MPIEXEC} -n 2 ./ex36 -levels 2 -nu .01 -n 2 -mg -ksp_converged_reason > ex36_2.tmp 2>&1;\
	   if (${DIFF} output/ex36_2.out ex36_2.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with runex36_2, diffs above \n========================================="; fi ;\
	   ${RM} -f ex36_2.tmp

runex42:
	-@${MPIEXEC} -n 1 ./ex42 -stokes_ksp_monitor_short -stokes_ksp_converged_reason -stokes_pc_type lu > ex42_1.tmp 2>&1;\
	   if (${DIFF} output/ex42_1.out ex42_1.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with runex42, diffs above \n========================================="; fi ;\
	   ${RM} -f ex42_1.tmp

runex42_2:
	-@${MPIEXEC} -n 3 ./ex42 -stokes_ksp_monitor_short -stokes_ksp_converged_reason -stokes_pc_type redundant -stokes_redundant_pc_type lu > ex42_2.tmp 2>&1;\
	   if (${DIFF} output/ex42_2.out ex42_2.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with runex42_2, diffs above \n========================================="; fi ;\
	   ${RM} -f ex42_2.tmp

runex43:
	-@${MPIEXEC} -n 1 ./ex43 -stokes_pc_use_amat -stokes_ksp_type fgmres -stokes_pc_type fieldsplit -stokes_pc_fieldsplit_block_size 3 -stokes_pc_fieldsplit_type SYMMETRIC_MULTIPLICATIVE -stokes_pc_fieldsplit_0_fields 0,1 -stokes_pc_fieldsplit_1_fields 2 -stokes_fieldsplit_0_ksp_type preonly -stokes_fieldsplit_0_pc_type lu -stokes_fieldsplit_1_ksp_type preonly -stokes_fieldsplit_1_pc_type jacobi -c_str 0 -solcx_eta0 1.0 -solcx_eta1 1.0e6 -solcx_xc 0.5 -solcx_nz 2 -mx 20 -my 20 -stokes_ksp_monitor_short > ex43_1.tmp 2>&1;	  \
	   ${DIFF} output/ex43_1.out ex43_1.tmp || printf "${PWD}\nPossible problem with ex43_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex43_1.tmp

runex43_2:
	-@${MPIEXEC} -n 1 ./ex43 -stokes_pc_use_amat -stokes_ksp_type fgmres -stokes_pc_type fieldsplit -stokes_pc_fieldsplit_block_size 3 -stokes_pc_fieldsplit_type SYMMETRIC_MULTIPLICATIVE -stokes_fieldsplit_u_ksp_type preonly -stokes_fieldsplit_u_pc_type lu -stokes_fieldsplit_p_ksp_type preonly -stokes_fieldsplit_p_pc_type jacobi -c_str 0 -solcx_eta0 1.0 -solcx_eta1 1.0e6 -solcx_xc 0.5 -solcx_nz 2 -mx 20 -my 20 -stokes_ksp_monitor_short > ex43_2.tmp 2>&1;	  \
	   ${DIFF} output/ex43_1.out ex43_2.tmp || printf "${PWD}\nPossible problem with ex43_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex43_2.tmp

runex43_3:
	-@${MPIEXEC} -n 4 ./ex43 -stokes_pc_use_amat -stokes_ksp_type gcr -stokes_ksp_gcr_restart 60 -stokes_ksp_norm_type unpreconditioned -stokes_ksp_rtol 1.e-2 -c_str 3 -sinker_eta0 1.0 -sinker_eta1 100 -sinker_dx 0.4 -sinker_dy 0.3 -mx 128 -my 128 -stokes_ksp_monitor_short -stokes_pc_type mg -stokes_mg_levels_pc_type fieldsplit -stokes_pc_use_amat false -stokes_pc_mg_galerkin pmat -stokes_mg_levels_pc_fieldsplit_block_size 3 -stokes_mg_levels_pc_fieldsplit_0_fields 0,1 -stokes_mg_levels_pc_fieldsplit_1_fields 2 -stokes_mg_levels_fieldsplit_0_pc_type sor -stokes_mg_levels_fieldsplit_1_pc_type sor -stokes_mg_levels_ksp_type chebyshev -stokes_mg_levels_ksp_max_it 1 -stokes_mg_levels_ksp_chebyshev_esteig 0,0.2,0,1.1 -stokes_pc_mg_levels 4 -stokes_ksp_view > ex43_3.tmp 2>&1;	  \
	   ${DIFF} output/ex43_3.out ex43_3.tmp || printf "${PWD}\nPossible problem with ex43_3, diffs above\n=========================================\n"; \
	   ${RM} -f ex43_3.tmp

runex43_bjacobi:
	-@${MPIEXEC} -n 4 ./ex43 -stokes_ksp_rtol 1.e-4 -stokes_pc_type bjacobi -stokes_pc_bjacobi_blocks 2 -dm_mat_type aij -stokes_ksp_converged_reason > ex43.tmp 2>&1;	  \
	   ${DIFF} output/ex43_bjacobi.out ex43.tmp || printf "${PWD}\nPossible problem with ex43_bjacobi, diffs above\n=========================================\n"; \
	   ${RM} -f ex43.tmp
runex43_bjacobi_baij:
	-@${MPIEXEC} -n 4 ./ex43 -stokes_ksp_rtol 1.e-4 -stokes_pc_type bjacobi -stokes_pc_bjacobi_blocks 2 -dm_mat_type baij -stokes_ksp_converged_reason > ex43.tmp 2>&1;	  \
	   ${DIFF} output/ex43_bjacobi.out ex43.tmp || printf "${PWD}\nPossible problem with ex43_bjacobi_baij, diffs above\n=========================================\n"; \
	   ${RM} -f ex43.tmp

runex43_nested_gmg:
	-@${MPIEXEC} -n 4 ./ex43 -stokes_pc_fieldsplit_off_diag_use_amat -mx 16 -my 16 -stokes_ksp_type fgmres  -stokes_pc_type fieldsplit -stokes_fieldsplit_u_pc_type mg -stokes_fieldsplit_u_pc_mg_levels 5 -stokes_fieldsplit_u_pc_mg_galerkin pmat -stokes_fieldsplit_u_ksp_type cg -stokes_fieldsplit_u_ksp_rtol 1.0e-4 -stokes_fieldsplit_u_mg_levels_pc_type jacobi -solcx_eta0 1.0e4 -stokes_fieldsplit_u_ksp_converged_reason -stokes_ksp_converged_reason -stokes_fieldsplit_p_sub_pc_factor_zeropivot 1.e-8 > ex43.tmp 2>&1;	  \
	   ${DIFF} output/ex43_nested_gmg.out ex43.tmp || printf "${PWD}\nPossible problem with ex43_nested_gmg, diffs above\n=========================================\n"; \
	   ${RM} -f ex43.tmp

runex43_4:
	-@${MPIEXEC} -n 4 ./ex43 -stokes_ksp_type pipegcr -stokes_ksp_pipegcr_mmax 60 -stokes_ksp_pipegcr_unroll_w 1 -stokes_ksp_norm_type natural -c_str 3 -sinker_eta0 1.0 -sinker_eta1 100 -sinker_dx 0.4 -sinker_dy 0.3 -mx 128 -my 128 -stokes_ksp_monitor_short -stokes_pc_type mg -stokes_mg_levels_pc_type fieldsplit  -stokes_pc_use_amat false  -stokes_pc_mg_galerkin pmat -stokes_mg_levels_pc_fieldsplit_block_size 3 -stokes_mg_levels_pc_fieldsplit_0_fields 0,1 -stokes_mg_levels_pc_fieldsplit_1_fields 2 -stokes_mg_levels_fieldsplit_0_pc_type sor -stokes_mg_levels_fieldsplit_1_pc_type sor -stokes_mg_levels_ksp_type chebyshev -stokes_mg_levels_ksp_max_it 1 -stokes_mg_levels_ksp_chebyshev_esteig 0,0.2,0,1.1 -stokes_pc_mg_levels 4 -stokes_ksp_view > ex43_4.tmp 2>&1;	  \
	   ${DIFF} output/ex43_4.out ex43_4.tmp || printf "${PWD}\nPossible problem with ex43_4, diffs above\n=========================================\n"; \
	   ${RM} -f ex43_4.tmp

runex43_5:
	-@${MPIEXEC} -n 4 ./ex43 -stokes_pc_fieldsplit_off_diag_use_amat -stokes_ksp_type pipegcr -stokes_pc_type fieldsplit -stokes_pc_fieldsplit_block_size 3 -stokes_pc_fieldsplit_type SYMMETRIC_MULTIPLICATIVE -stokes_pc_fieldsplit_0_fields 0,1 -stokes_pc_fieldsplit_1_fields 2 -stokes_fieldsplit_0_ksp_type preonly -stokes_fieldsplit_0_pc_type bjacobi -stokes_fieldsplit_1_ksp_type preonly -stokes_fieldsplit_1_pc_type bjacobi -c_str 0 -solcx_eta0 1.0 -solcx_eta1 1.0e6 -solcx_xc 0.5 -solcx_nz 2 -mx 20 -my 20 -stokes_ksp_monitor_short -stokes_ksp_view > ex43_5.tmp 2>&1;	  \
	   ${DIFF} output/ex43_5.out ex43_5.tmp || printf "${PWD}\nPossible problem with ex43_5, diffs above\n=========================================\n"; \
	   ${RM} -f ex43_5.tmp

runex43_6:
	-@${MPIEXEC} -n 8 ./ex43 -stokes_ksp_view -stokes_pc_type mg -stokes_pc_mg_levels 2 -stokes_mg_coarse_pc_type telescope -stokes_mg_coarse_pc_telescope_reduction_factor 2  -stokes_pc_use_amat false -stokes_pc_mg_galerkin pmat -stokes_mg_coarse_pc_telescope_subcomm_type contiguous > ex43_6.tmp 2>&1; \
	   ${DIFF} output/ex43_6.out ex43_6.tmp || printf "${PWD}\nPossible problem with ex43_6, diffs above\n=========================================\n"; \
	   ${RM} -f ex43_6.tmp

runex45:
	-@${MPIEXEC} -n 4 ./ex45 -pc_type exotic -ksp_monitor_short -ksp_type fgmres -mg_levels_ksp_type gmres -mg_levels_ksp_max_it 1 -mg_levels_pc_type bjacobi > ex45_1.tmp 2>&1;	  \
	   ${DIFF} output/ex45_1.out ex45_1.tmp || printf "${PWD}\nPossible problem with ex45_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex45_1.tmp
runex45_2:
	-@${MPIEXEC} -n 4 ./ex45 -ksp_monitor_short -da_grid_x 21 -da_grid_y 21 -da_grid_z 21 -pc_type mg -pc_mg_levels 3 -mg_levels_ksp_type richardson -mg_levels_ksp_max_it 1 -mg_levels_pc_type bjacobi > ex45_2.tmp 2>&1; \
	   ${DIFF} output/ex45_2.out ex45_2.tmp || printf "${PWD}\nPossible problem with ex45_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex45_2.tmp

runex45_telescope:
	-@${MPIEXEC} -n 4 ./ex45 -ksp_type fgmres -ksp_monitor_short -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type jacobi  -pc_mg_levels 2 -da_grid_x 65 -da_grid_y 65 -da_grid_z 65 -mg_coarse_pc_type telescope -mg_coarse_pc_telescope_ignore_kspcomputeoperators -mg_coarse_pc_telescope_reduction_factor 4 -mg_coarse_telescope_pc_type mg -mg_coarse_telescope_pc_mg_galerkin pmat -mg_coarse_telescope_pc_mg_levels 3 -mg_coarse_telescope_mg_levels_ksp_type richardson -mg_coarse_telescope_mg_levels_pc_type jacobi   -mg_levels_ksp_type richardson -mg_coarse_telescope_mg_levels_ksp_type richardson -ksp_rtol 1.0e-4  > ex45_telescope.tmp 2>&1; \
	   ${DIFF} output/ex45_telescope.out ex45_telescope.tmp || printf "${PWD}\nPossible problem with ex45_telescope, diffs above\n=========================================\n"; \
	   ${RM} -f ex45_telescope.tmp

runex45_telescope_2:
	-@${MPIEXEC} -n 4 ./ex45 -ksp_type fgmres -ksp_monitor_short -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type jacobi  -pc_mg_levels 2 -da_grid_x 65 -da_grid_y 65 -da_grid_z 65 -mg_coarse_pc_type telescope -mg_coarse_pc_telescope_reduction_factor 2 -mg_coarse_telescope_pc_type mg -mg_coarse_telescope_pc_mg_galerkin pmat -mg_coarse_telescope_pc_mg_levels 3 -mg_coarse_telescope_mg_levels_ksp_type richardson -mg_coarse_telescope_mg_levels_pc_type jacobi   -mg_levels_ksp_type richardson -mg_coarse_telescope_mg_levels_ksp_type richardson -ksp_rtol 1.0e-4  > ex45_telescope_2.tmp 2>&1; \
	   ${DIFF} output/ex45_telescope_2.out ex45_telescope_2.tmp || printf "${PWD}\nPossible problem with ex45_telescope_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex45_telescope_2.tmp

runex44f:
	-@${MPIEXEC} -n 1 ./ex44f -ksp_converged_reason > ex44f_1.tmp 2>&1;\
	   if (${DIFF} output/ex44f_1.out ex44f_1.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with runex44f, diffs above \n========================================="; fi ;\
	   ${RM} -f ex44f_1.tmp

runex45f:
	-@${MPIEXEC} -n 4 ./ex45f -ksp_monitor_short -da_refine 5 -pc_type mg -pc_mg_levels 5 -mg_levels_ksp_type chebyshev -mg_levels_ksp_max_it 2 -mg_levels_pc_type jacobi -ksp_pc_side right > ex45f_1.tmp 2>&1; \
	   ${DIFF} output/ex45f_1.out ex45f_1.tmp || printf "${PWD}\nPossible problem with ex45f_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex45f_1.tmp

runex46_aijcusp:
	-@${MPIEXEC} -n 1 ./ex46 -dm_mat_type aijcusp -dm_vec_type cusp -random_exact_sol > ex46_aijcusp.tmp 2>& 1; \
		${DIFF} output/ex46_aijcusp.out ex46_aijcusp.tmp || printf "${PWD}\nPossible problem with ex46_aijcusp, diffs above\n=========================================\n"; \
		${RM} ex46_aijcusp.tmp
runex46_aijcusparse:
	-@${MPIEXEC} -n 1 ./ex46 -dm_mat_type aijcusparse -dm_vec_type cuda -random_exact_sol -pc_type ilu -pc_factor_mat_solver_package cusparse > ex46_aijcusparse.tmp 2>& 1; \
		${DIFF} output/ex46_aijcusparse.out ex46_aijcusparse.tmp || printf "${PWD}\nPossible problem with ex46_aijcusparse, diffs above\n=========================================\n"; \
		${RM} ex46_aijcusparse.tmp

runex49:
	-@${MPIEXEC} -n 1 ./ex49 -mx 20 -my 30 -elas_ksp_monitor_short -no_view -c_str 3 -sponge_E0 1 -sponge_E1 1000 -sponge_nu0 0.4 -sponge_nu1 0.2 -sponge_t 1 -sponge_w 8 -elas_ksp_rtol 5e-3 -elas_ksp_view  > ex49_1.tmp 2>&1;	  \
	   ${DIFF} output/ex49_1.out ex49_1.tmp || printf "${PWD}\nPossible problem with ex49_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex49_1.tmp

runex49_2:
	-@${MPIEXEC} -n 4 ./ex49 -mx 20 -my 30 -elas_ksp_monitor_short -no_view -c_str 3 -sponge_E0 1 -sponge_E1 1000 -sponge_nu0 0.4 -sponge_nu1 0.2 -sponge_t 1 -sponge_w 8 -elas_ksp_type gcr -elas_pc_type asm -elas_sub_pc_type lu -elas_ksp_rtol 5e-3 > ex49_2.tmp 2>&1;	  \
	   ${DIFF} output/ex49_2.out ex49_2.tmp || printf "${PWD}\nPossible problem with ex49_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex49_2.tmp

runex49_3:
	-@${MPIEXEC} -n 4 ./ex49 -mx 20 -my 30 -elas_ksp_monitor_short -no_view -c_str 2 -brick_E 1,10,1000,100 -brick_nu 0.4,0.2,0.3,0.1 -brick_span 3 -elas_pc_type asm -elas_sub_pc_type lu -elas_ksp_rtol 5e-3  > ex49_3.tmp 2>&1; \
	   ${DIFF} output/ex49_3.out ex49_3.tmp || printf "${PWD}\nPossible problem with ex49_3, diffs above\n=========================================\n"; \
	   ${RM} -f ex49_3.tmp

runex49_4:
	-@${MPIEXEC} -n 4 ./ex49 -elas_ksp_monitor_short -elas_ksp_converged_reason -elas_ksp_type cg -elas_ksp_norm_type unpreconditioned -mx 40 -my 40 -c_str 2 -brick_E 1,1e-6,1e-2 -brick_nu .3,.2,.4 -brick_span 8 -elas_mg_levels_ksp_type chebyshev -elas_pc_type ml -elas_mg_levels_ksp_chebyshev_esteig 0,0.2,0,1.1 -elas_mg_levels_pc_type pbjacobi -elas_mg_levels_ksp_max_it 2 -use_nonsymbc -elas_pc_ml_nullspace user > ex49_4.tmp 2>&1; \
	   ${DIFF} output/ex49_4.out ex49_4.tmp || printf "${PWD}\nPossible problem with ex49_4, diffs above\n=========================================\n"; \
	   ${RM} -f ex49_4.tmp mesh-p0*.dat properties-p0*.dat X-p*.dat

runex49_5:
	-@${MPIEXEC} -n 3 ./ex49 -elas_ksp_monitor_short -elas_ksp_converged_reason -elas_ksp_type cg -elas_ksp_norm_type natural -mx 22 -my 22 -c_str 2 -brick_E 1,1e-6,1e-2 -brick_nu .3,.2,.4 -brick_span 8 -elas_pc_type gamg -elas_mg_levels_ksp_type chebyshev -elas_mg_levels_ksp_max_it 1 -elas_mg_levels_ksp_chebyshev_esteig 0.2,1.1 -elas_mg_levels_pc_type jacobi > ex49_5.tmp 2>&1; \
	   ${DIFF} output/ex49_5.out ex49_5.tmp || printf "${PWD}\nPossible problem with ex49_5, diffs above\n=========================================\n"; \
	   ${RM} -f ex49_5.tmp mesh-p0*.dat properties-p0*.dat X-p*.dat

# hyper has some valgrind serious bus in it for vec_interp_variant hence this crashes on some systems, waiting for hypre team to fix
runex49_hypre_nullspace:
	-@${MPIEXEC} -n 1 ./ex49 -elas_ksp_monitor_short -elas_ksp_converged_reason -elas_ksp_type cg -elas_ksp_norm_type natural -mx 22 -my 22 -c_str 2 -brick_E 1,1e-6,1e-2 -brick_nu .3,.2,.4 -brick_span 8 -elas_pc_type hypre  -elas_pc_hypre_boomeramg_nodal_coarsen  6 -elas_pc_hypre_boomeramg_vec_interp_variant 3 -elas_ksp_view > ex49_hypre_nullspace.tmp 2>&1; \
	   ${DIFF} output/ex49_hypre_nullspace.out ex49_hypre_nullspace.tmp || printf "${PWD}\nPossible problem with ex49_hypre_nullspace, diffs above\n=========================================\n"; \
	   ${RM} -f ex49_hypre_nullspace.tmp

runex49_6:
	-@${MPIEXEC} -n 4 ./ex49 -mx 20 -my 30 -elas_ksp_monitor_short -no_view -c_str 3 -sponge_E0 1 -sponge_E1 1000 -sponge_nu0 0.4 -sponge_nu1 0.2 -sponge_t 1 -sponge_w 8 -elas_ksp_type pipegcr -elas_pc_type asm -elas_sub_pc_type lu > ex49_6.tmp 2>&1;	  \
	   ${DIFF} output/ex49_6.out ex49_6.tmp || printf "${PWD}\nPossible problem with ex49_6, diffs above\n=========================================\n"; \
	   ${RM} -f ex49_6.tmp

runex49_7:
	-@${MPIEXEC} -n 4 ./ex49 -mx 20 -my 30 -elas_ksp_monitor_short -no_view -c_str 3 -sponge_E0 1 -sponge_E1 1000 -sponge_nu0 0.4 -sponge_nu1 0.2 -sponge_t 1 -sponge_w 8 -elas_ksp_type pipegcr -elas_pc_type asm -elas_sub_pc_type ksp -elas_sub_ksp_ksp_type cg -elas_sub_ksp_ksp_max_it 15 > ex49_7.tmp 2>&1;	  \
	   ${DIFF} output/ex49_7.out ex49_7.tmp || printf "${PWD}\nPossible problem with ex49_7, diffs above\n=========================================\n"; \
	   ${RM} -f ex49_7.tmp

runex49_8:
	-@${MPIEXEC} -n 4 ./ex49 -mx 20 -my 30 -elas_ksp_monitor_short -no_view -c_str 3 -sponge_E0 1 -sponge_E1 1000 -sponge_nu0 0.4 -sponge_nu1 0.2 -sponge_t 1 -sponge_w 8 -elas_ksp_type pipefgmres -elas_pc_type asm -elas_sub_pc_type ksp -elas_sub_ksp_ksp_type cg -elas_sub_ksp_ksp_max_it 15 > ex49_8.tmp 2>&1;	  \
	   ${DIFF} output/ex49_8.out ex49_8.tmp || printf "${PWD}\nPossible problem with ex49_8, diffs above\n=========================================\n"; \
	   ${RM} -f ex49_8.tmp

runex50:
	-@${MPIEXEC} -n 1 ./ex50 -pc_type mg -pc_mg_type full -ksp_type cg -ksp_monitor_short -da_refine 3  -mg_coarse_pc_type svd -ksp_view  > ex50.tmp 2>&1;         \
        ${DIFF} output/ex50_1.out ex50.tmp || printf "${PWD}\nPossible problem with ex50, diffs above\n=========================================\n"; \
        ${RM} -f ex50.tmp

runex50_2:
	-@${MPIEXEC} -n 4 ./ex50 -pc_type mg -pc_mg_type full -ksp_type cg -ksp_monitor_short -da_refine 3  -mg_coarse_pc_type redundant -mg_coarse_redundant_pc_type svd -ksp_view > ex50_2.tmp 2>&1;         \
        ${DIFF} output/ex50_2.out ex50_2.tmp || printf "${PWD}\nPossible problem with ex50_2, diffs above\n=========================================\n"; \
        ${RM} -f ex50_2.tmp

runex50_3 :
	-@${MPIEXEC} -n 2 ./ex50 -pc_type mg -pc_mg_type full -ksp_monitor_short -da_refine 5 -mg_coarse_ksp_type cg -mg_coarse_ksp_converged_reason -mg_coarse_ksp_rtol 1e-2 -mg_coarse_ksp_max_it 5 -mg_coarse_pc_type none -pc_mg_levels 2 -ksp_type pipefgmres -ksp_pipefgmres_shift 1.5 > ex50_3.tmp 2>&1;	  \
	   ${DIFF} output/ex50_3.out ex50_3.tmp || printf "${PWD}\nPossible problem with ex50_3, diffs above\n=========================================\n"; \
	   ${RM} -f ex50_3.tmp

runex51:
	-@${MPIEXEC} -n 2 ./ex51 -ksp_monitor_short > ex51.tmp 2>&1;	  \
	   ${DIFF} output/ex51_1.out ex51.tmp || printf "${PWD}\nPossible problem with ex51, diffs above\n=========================================\n"; \
	   ${RM} -f ex51.tmp

runex52:
	-@${MPIEXEC} -n 1 ./ex52 -use_petsc_lu > ex52.tmp 2>&1;	  \
	   ${DIFF} output/ex52_2.out ex52.tmp || printf "${PWD}\nPossible problem with ex52, diffs above\n=========================================\n"; \
	   ${RM} -f ex52.tmp
runex52_mumps:
	-@${MPIEXEC} -n 3 ./ex52 -use_mumps_lu > ex52.tmp 2>&1;	  \
	   ${DIFF} output/ex52_1.out ex52.tmp || printf "${PWD}\nPossible problem with ex52_mumps, diffs above\n=========================================\n"; \
	   ${RM} -f ex52.tmp
runex52_mumps_2:
	-@${MPIEXEC} -n 3 ./ex52 -use_mumps_ch > ex52.tmp 2>&1;	  \
	   ${DIFF} output/ex52_1.out ex52.tmp || printf "${PWD}\nPossible problem with ex52_mumps_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex52.tmp
runex52_mumps_3:
	-@${MPIEXEC} -n 3 ./ex52 -use_mumps_ch -mat_type sbaij > ex52.tmp 2>&1;	  \
	   ${DIFF} output/ex52_1.out ex52.tmp || printf "${PWD}\nPossible problem with ex52_mumps_3, diffs above\n=========================================\n"; \
	   ${RM} -f ex52.tmp
runex52_superlu_ilu:
	-@${MPIEXEC} -n 1 ./ex52 -use_superlu_ilu > ex52.tmp 2>&1;	  \
	   ${DIFF} output/ex52_2.out ex52.tmp || printf "${PWD}\nPossible problem with ex52_superlu_ilu, diffs above\n=========================================\n"; \
	   ${RM} -f ex52.tmp
runex52_superlu:
	-@${MPIEXEC} -n 1 ./ex52 -use_superlu_lu > ex52.tmp 2>&1;	  \
	   ${DIFF} output/ex52_2.out ex52.tmp || printf "${PWD}\nPossible problem with ex52_superlu, diffs above\n=========================================\n"; \
	   ${RM} -f ex52.tmp
runex52_superlu_dist:
	-@${MPIEXEC} -n 2 ./ex52 -use_superlu_lu > ex52.tmp 2>&1;	  \
	   ${DIFF} output/ex52_2.out ex52.tmp || printf "${PWD}\nPossible problem with ex52_superlu_dist, diffs above\n=========================================\n"; \
	   ${RM} -f ex52.tmp
runex52_strumpack_ilu:
	-@${MPIEXEC} -n 1 ./ex52 -use_strumpack_ilu 2>&1 > ex52.tmp;	  \
	   ${DIFF} output/ex52_3.out ex52.tmp || printf "${PWD}\nPossible problem with ex52_strumpack_ilu, diffs above\n=========================================\n"; \
	   ${RM} -f ex52.tmp
runex52_strumpack:
	-@${MPIEXEC} -n 1 ./ex52 -use_strumpack_lu 2>&1 > ex52.tmp;	  \
	   ${DIFF} output/ex52_3.out ex52.tmp || printf "${PWD}\nPossible problem with ex52_strumpack, diffs above\n=========================================\n"; \
	   ${RM} -f ex52.tmp
runex52_strumpack_ilu_2:
	-@${MPIEXEC} -n 2 ./ex52 -use_strumpack_ilu 2>&1 > ex52.tmp ;	  \
	   ${DIFF} output/ex52_3.out ex52.tmp || printf "${PWD}\nPossible problem with ex52_strumpack_ilu, diffs above\n=========================================\n"; \
	   ${RM} -f ex52.tmp
runex52_strumpack_2:
	-@${MPIEXEC} -n 2 ./ex52 -use_strumpack_lu 2>&1 > ex52.tmp ;	  \
	   ${DIFF} output/ex52_3.out ex52.tmp || printf "${PWD}\nPossible problem with ex52_strumpack, diffs above\n=========================================\n"; \
	   ${RM} -f ex52.tmp

runex52f_mumps:
	-@${MPIEXEC} -n 3 ./ex52f > ex52.tmp 2>&1;	  \
	   ${DIFF} output/ex52f_1.out ex52.tmp || printf "${PWD}\nPossible problem with ex52f_mumps, diffs above\n=========================================\n"; \
	   ${RM} -f ex52.tmp

runex53:
	-@${MPIEXEC} -n 1 ./ex53 > ex53.tmp 2>&1;         \
        ${DIFF} output/ex53.out ex53.tmp || printf "${PWD}\nPossible problem with ex53, diffs above\n=========================================\n"; \
        ${RM} -f ex53.tmp

runex53_2:
	-@${MPIEXEC} -n 2 ./ex53 > ex53.tmp 2>&1;	  \
	   ${DIFF} output/ex53.out ex53.tmp || printf "${PWD}\nPossible problem with ex53_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex53.tmp

runex54_geo:
	-@${MPIEXEC} -n 4 ./ex54 -ne 49 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_type geo -pc_gamg_coarse_eq_limit 200 -mg_levels_pc_type jacobi -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.05 -ksp_monitor_short -mg_levels_esteig_ksp_type cg > ex.tmp 2>&1;	  \
         ${DIFF} output/ex54_0.out ex.tmp || printf "${PWD}\nPossible problem with ex54_0.out, diffs above\n=========================================\n"; \
        ${RM} -f ex.tmp

runex54:
	-@${MPIEXEC} -n 4 ./ex54 -ne 49 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 1 -ksp_converged_reason -mg_levels_esteig_ksp_type cg > ex.tmp 2>&1; \
         ${DIFF} output/ex54_1.out ex.tmp || printf "${PWD}\nPossible problem with ex54_1.out, diffs above\n======================================\n"; \
        ${RM} -f ex.tmp

runex54_Classical:
	-@${MPIEXEC} -n 1 ./ex54 -ne 49 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_type classical -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.05 -ksp_converged_reason -mg_levels_esteig_ksp_type cg > ex.tmp 2>&1; \
         ${DIFF} output/ex54_classical.out ex.tmp || printf "${PWD}\nPossible problem with ex54_classical.out, diffs above\n======================================\n"; \
       ${RM} -f ex.tmp

runex54f:
	-@x="bad"; ${MPIEXEC}  -n 4 ./ex54f -ne 39 -theta 30.0 -epsilon 1.e-1 -blob_center 0.,0. -ksp_type cg -pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 1 -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.05 -mat_coarsen_type hem -pc_gamg_square_graph 0 -ksp_monitor_short -mg_levels_esteig_ksp_type cg  > ex54f_1.tmp 2>&1; \
	   if (${DIFF} output/ex54f.out ex54f_1.tmp > /dev/null 2>&1) then x='good'; fi ;\
	   if (${DIFF} output/ex54f_1_alt.out ex54f_1.tmp > /dev/null 2>&1) then x='good'; fi; \
	   if [ "$$x" = "bad" ]; then ${DIFF} output/ex54f.out ex54f_1.tmp ; ${DIFF} output/ex54f_1_alt.out ex54f_1.tmp ; printf "${PWD}\nPossible problem with ex54f_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex54f_1.tmp

runex55_geo:
	-@${MPIEXEC} -n 4 ./ex55 -ne 29 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_type geo -use_coordinates -ksp_monitor_short -mg_levels_esteig_ksp_type cg -ksp_type cg -ksp_norm_type unpreconditioned > ex.tmp 2>&1;	  \
         ${DIFF} output/ex55_0.out ex.tmp || printf "${PWD}\nPossible problem with ex55_0, diffs above\n=========================================\n"; \
        ${RM} -f ex.tmp

runex55_hypre:
	-@${MPIEXEC} -n 4 ./ex55 -ne 29 -alpha 1.e-3 -ksp_type cg -pc_type hypre -pc_hypre_type boomeramg -ksp_monitor_short > ex.tmp 2>&1;	  \
         ${DIFF} output/ex55_hypre.out ex.tmp || printf "${PWD}\nPossible problem with ex55_hypre, diffs above\n=========================================\n"; \
        ${RM} -f ex.tmp

runex55:
	-@${MPIEXEC} -n 4 ./ex55 -ne 29 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 1 -use_coordinates -ksp_converged_reason -mg_levels_esteig_ksp_type cg -ksp_rtol 1.e-3 -ksp_monitor_short > ex.tmp 2>&1; \
         ${DIFF} output/ex55_sa.out ex.tmp || printf "${PWD}\nPossible problem with ex55_sa, diffs above\n=========================================\n"; \
        ${RM} -f ex.tmp

runex55_Classical:
	-@${MPIEXEC} -n 4 ./ex55 -ne 29 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_type classical -mg_levels_ksp_max_it 5 -ksp_converged_reason -mg_levels_esteig_ksp_type cg > ex.tmp 2>&1; \
         ${DIFF} output/ex55_classical.out ex.tmp || printf "${PWD}\nPossible problem with ex55_classical, diffs above\n=========================================\n"; \
        ${RM} -f ex.tmp

runex55_NC:
	-@${MPIEXEC} -n 4 ./ex55 -ne 29 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 1 -ksp_converged_reason -mg_levels_esteig_ksp_type cg > ex.tmp 2>&1; \
         ${DIFF} output/ex55_NC.out ex.tmp || printf "${PWD}\nPossible problem with ex55_NC, diffs above\n======================================\n"; \
        ${RM} -f ex.tmp

runex56:
	-@${MPIEXEC} -n 8 ./ex56 -ne 13 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_agg_nsmooths 1 -pc_gamg_reuse_interpolation true -two_solves -ksp_converged_reason -ksp_view -use_mat_nearnullspace -mg_levels_esteig_ksp_type cg -mg_levels_esteig_ksp_max_it 10 -pc_gamg_square_graph 1 -mg_levels_ksp_max_it 1 -mg_levels_ksp_type chebyshev -mg_levels_ksp_chebyshev_esteig 0,0.2,0,1.05 -gamg_est_ksp_type cg -gamg_est_ksp_max_it 10 -pc_gamg_asm_use_agg true -mg_levels_sub_pc_type lu -mg_levels_pc_asm_overlap 0 -pc_gamg_threshold -0.01 -pc_gamg_coarse_eq_limit 200 -pc_gamg_process_eq_limit 30 -pc_gamg_repartition false -pc_mg_cycle_type v -pc_gamg_use_parallel_coarse_grid_solver -mg_coarse_pc_type jacobi -mg_coarse_ksp_type cg -ksp_monitor_short | grep -v variant > ex.tmp 2>&1; \
         ${DIFF} output/ex56_1.out ex.tmp || printf "${PWD}\nPossible problem with ex56, diffs above \n=========================================\n"; \
         ${RM} -f ex.tmp

runex56_2:
	-@${MPIEXEC} -n 8 ./ex56 -ne 31 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_agg_nsmooths 1 -pc_gamg_reuse_interpolation true -two_solves -ksp_converged_reason -use_mat_nearnullspace -mg_levels_esteig_ksp_type cg -mg_levels_esteig_ksp_max_it 10 -pc_gamg_square_graph 1 -mg_levels_ksp_max_it 1 -mg_levels_ksp_type chebyshev -mg_levels_ksp_chebyshev_esteig 0,0.2,0,1.05 -gamg_est_ksp_type cg -gamg_est_ksp_max_it 10 -pc_gamg_asm_use_agg true -mg_levels_sub_pc_type lu -mg_levels_pc_asm_overlap 0 -pc_gamg_threshold -0.01 -pc_gamg_coarse_eq_limit 200 -pc_gamg_process_eq_limit 30 -pc_gamg_repartition false -pc_mg_cycle_type v -pc_gamg_use_parallel_coarse_grid_solver -mg_coarse_pc_type jacobi -mg_coarse_ksp_type cg | grep -v variant > ex.tmp 2>&1; \
         ${DIFF} output/ex56_2.out ex.tmp || printf "${PWD}\nPossible problem with ex56_2, diffs above \n=========================================\n"; \
         ${RM} -f ex.tmp

runex56_latebs:
	-@${MPIEXEC} -n 8 ./ex56 -test_late_bs 0 -ne 13 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_agg_nsmooths 1 -pc_gamg_reuse_interpolation true -two_solves -ksp_converged_reason -ksp_view -use_mat_nearnullspace -mg_levels_esteig_ksp_type cg -mg_levels_esteig_ksp_max_it 10 -pc_gamg_square_graph 1 -mg_levels_ksp_max_it 1 -mg_levels_ksp_type chebyshev -mg_levels_ksp_chebyshev_esteig 0,0.2,0,1.05 -gamg_est_ksp_type cg -gamg_est_ksp_max_it 10 -pc_gamg_asm_use_agg true -mg_levels_sub_pc_type lu -mg_levels_pc_asm_overlap 0 -pc_gamg_threshold -0.01 -pc_gamg_coarse_eq_limit 200 -pc_gamg_process_eq_limit 30 -pc_gamg_repartition false -pc_mg_cycle_type v -pc_gamg_use_parallel_coarse_grid_solver -mg_coarse_pc_type jacobi -mg_coarse_ksp_type cg -ksp_monitor_short -ksp_view | grep -v variant > ex.tmp 2>&1; \
	 ${MPIEXEC} -n 8 ./ex56 -test_late_bs -ne 13 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_agg_nsmooths 1 -pc_gamg_reuse_interpolation true -two_solves -ksp_converged_reason -ksp_view -use_mat_nearnullspace -mg_levels_esteig_ksp_type cg -mg_levels_esteig_ksp_max_it 10 -pc_gamg_square_graph 1 -mg_levels_ksp_max_it 1 -mg_levels_ksp_type chebyshev -mg_levels_ksp_chebyshev_esteig 0,0.2,0,1.05 -gamg_est_ksp_type cg -gamg_est_ksp_max_it 10 -pc_gamg_asm_use_agg true -mg_levels_sub_pc_type lu -mg_levels_pc_asm_overlap 0 -pc_gamg_threshold -0.01 -pc_gamg_coarse_eq_limit 200 -pc_gamg_process_eq_limit 30 -pc_gamg_repartition false -pc_mg_cycle_type v -pc_gamg_use_parallel_coarse_grid_solver -mg_coarse_pc_type jacobi -mg_coarse_ksp_type cg -ksp_monitor_short -ksp_view | grep -v variant > exlate.tmp 2>&1; \
         ${DIFF} exlate.tmp ex.tmp || printf "${PWD}\nPossible problem with ex56_latebs, diffs above \n=========================================\n"; \
         ${RM} -f ex.tmp exlate.tmp


runex56_ml:
	-@${MPIEXEC} -n 8 ./ex56 -ne 9 -alpha 1.e-3 -ksp_type cg -pc_type ml -mg_levels_ksp_type chebyshev -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.05 -mg_levels_pc_type sor -ksp_monitor_short -mg_levels_esteig_ksp_type cg > ex.tmp 2>&1;	\
         ${DIFF} output/ex56_ml.out ex.tmp || printf "${PWD}\nPossible problem with ex56_ml, diffs above\n=========================================\n"; \
        ${RM} -f ex.tmp

runex56_nns:
	-@${MPIEXEC} -n 1 ./ex56 -ne 9 -alpha 1.e-3 -ksp_converged_reason -ksp_type cg -ksp_max_it 50 -pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 1 -pc_gamg_coarse_eq_limit 1000 -mg_levels_ksp_type chebyshev -mg_levels_pc_type sor -pc_gamg_reuse_interpolation true -two_solves -use_mat_nearnullspace -mg_levels_esteig_ksp_type cg > ex.tmp 2>&1;	\
         ${DIFF} output/ex56_nns.out ex.tmp || printf "${PWD}\nPossible problem with ex56_nns, diffs above\n=========================================\n"; \
         ${RM} -f ex.tmp

runex56_nns_telescope:
	-@${MPIEXEC} -n 2 ./ex56 -use_mat_nearnullspace -ksp_monitor_short -pc_type telescope -pc_telescope_reduction_factor 2 -telescope_pc_type gamg > ex.tmp 2>&1;	\
         ${DIFF} output/ex56_nns_telescope.out ex.tmp || printf "${PWD}\nPossible problem with ex56_nns_telescope, diffs above\n=========================================\n"; \
         ${RM} -f ex.tmp

runex58:
	-@${MPIEXEC} -n 1 ./ex58 -mat_type aij > ex58.tmp 2>&1;         \
	${DIFF} output/ex58.out ex58.tmp || printf "${PWD}\nPossible problem with ex58, diffs above\n=========================================\n"; \
	${RM} -f ex58.tmp
runex58_baij:
	-@${MPIEXEC} -n 1 ./ex58 -mat_type baij > ex58.tmp 2>&1;         \
	${DIFF} output/ex58.out ex58.tmp || printf "${PWD}\nPossible problem with ex58_baij, diffs above\n=========================================\n"; \
	${RM} -f ex58.tmp
runex58_sbaij:
	-@${MPIEXEC} -n 1 ./ex58 -mat_type sbaij > ex58.tmp 2>&1;         \
	${DIFF} output/ex58.out ex58.tmp || printf "${PWD}\nPossible problem with ex58_sbaij, diffs above\n=========================================\n"; \
	${RM} -f ex58.tmp

runex60:
	-@${MPIEXEC} -n 2 ./ex60 -ksp_monitor_short -ksp_rtol 1e-6 -diagfunc 1 -ksp_type fcg -ksp_fcg_mmax 1 -eta 0.1 > ex60_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex60_1.out ex60_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with with ex60_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex60_1.tmp
runex60_2:
	-@${MPIEXEC} -n 2 ./ex60 -ksp_monitor_short -diagfunc 3 -ksp_type fcg -ksp_fcg_mmax 10000 -eta 0.3333 > ex60_2.tmp 2>&1;	  \
	   if (${DIFF} output/ex60_2.out ex60_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with with ex60_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex60_2.tmp

runex60_3:
	-@${MPIEXEC} -n 3 ./ex60 -ksp_monitor_short -ksp_rtol 1e-6 -diagfunc 2 -ksp_type fgmres -eta 0.1 > ex60_3.tmp 2>&1;	  \
	   if (${DIFF} output/ex60_3.out ex60_3.tmp) then true; \
	   else printf "${PWD}\nPossible problem with with ex60_3, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex60_3.tmp

runex60_4:
	-@${MPIEXEC} -n 2 ./ex60 -ksp_monitor_short -ksp_rtol 1e-6 -diagfunc 1 -ksp_type pipefcg -ksp_pipefcg_mmax 1 -eta 0.1 > ex60_4.tmp 2>&1;	  \
	   if (${DIFF} output/ex60_4.out ex60_4.tmp) then true; \
	   else printf "${PWD}\nPossible problem with with ex60_4, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex60_4.tmp

runex60_5:
	-@${MPIEXEC} -n 2 ./ex60 -ksp_monitor_short -ksp_rtol 1e-6 -diagfunc 3 -ksp_type pipefcg -ksp_pipefcg_mmax 10000 -eta 0.1 > ex60_5.tmp 2>&1;	  \
           if (${DIFF} output/ex60_5.out ex60_5.tmp) then true; \
	   else printf "${PWD}\nPossible problem with with ex60_5, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex60_5.tmp

runex60_6 :
	-@${MPIEXEC} -n 4 ./ex60 -ksp_monitor_short -ksp_rtol 1e-6 -diagfunc 3 -ksp_type fcg -ksp_fcg_mmax 10000 -eta 0 -pc_type ksp -ksp_ksp_type cg -ksp_pc_type none -ksp_ksp_rtol 1e-1 -ksp_ksp_max_it 5 -ksp_ksp_converged_reason > ex60_6.tmp 2>&1; \
           if (${DIFF} output/ex60_6.out ex60_6.tmp) then true; \
	   else printf "${PWD}\nPossible problem with with ex60_6, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex60_6.tmp

runex60_7 :
	-@${MPIEXEC} -n 4 ./ex60 -ksp_monitor_short -ksp_rtol 1e-6 -diagfunc 3 -ksp_type pipefcg -ksp_pipefcg_mmax 10000 -eta 0 -pc_type ksp -ksp_ksp_type cg -ksp_pc_type none -ksp_ksp_rtol 1e-1 -ksp_ksp_max_it 5 -ksp_ksp_converged_reason > ex60_7.tmp 2>&1; \
           if (${DIFF} output/ex60_7.out ex60_7.tmp) then true; \
	   else printf "${PWD}\nPossible problem with with ex60_7, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex60_7.tmp

runex60_8 :
	-@${MPIEXEC} -n 2 ./ex60 -ksp_monitor_short -ksp_rtol 1e-6 -diagfunc 1 -ksp_type pipefgmres -pc_type ksp -ksp_ksp_type cg -ksp_pc_type none -ksp_ksp_rtol 1e-2 -ksp_ksp_converged_reason > ex60_8.tmp 2>&1;	  \
	   if (${DIFF} output/ex60_8.out ex60_8.tmp) then true; \
	   else printf "${PWD}\nPossible problem with with ex60_8, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex60_8.tmp

runex60_9 :
	-@${MPIEXEC} -n 2 ./ex60 -ksp_monitor_short -ksp_rtol 1e-6 -diagfunc 1 -ksp_type pipefgmres -pc_type ksp -ksp_ksp_type cg -ksp_pc_type none -ksp_ksp_rtol 1e-2 -ksp_ksp_converged_reason > ex60_9.tmp 2>&1;	  \
	   if (${DIFF} output/ex60_9.out ex60_9.tmp) then true; \
	   else printf "${PWD}\nPossible problem with with ex60_9, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex60_9.tmp

runex61f:
	-@${MPIEXEC} -n 1 ./ex61f > ex61f.tmp 2>&1; \
            if (${DIFF} output/ex61f_1.out ex61f.tmp) then true; \
            else printf "${PWD}\nPossible problem with ex61f, diffs above\n==========================\
===============\n"; fi; \
            ${RM} -f ex61f.tmp

NP = 1
M  = 4
N  = 5
MDOMAINS = 2
NDOMAINS = 1
OVERLAP=1

runex62_hp:
	-@${MPIEXEC} -n 4 ./ex62  -M 7  -N 9  -pc_gasm_overlap  1  -sub_pc_type lu  -sub_pc_factor_mat_solver_package superlu_dist  -ksp_monitor -print_error  -pc_gasm_total_subdomains 2 -pc_gasm_use_hierachical_partitioning 1 > ex62.tmp 2>&1;	  \
	   if (${DIFF} output/ex62.out ex62.tmp) then true; \
	   else printf "${PWD}\nPossible problem with with ex62, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex62.tmp

runex62_2D_1:
	-@${MPIEXEC} -n 1 ./ex62 -M 7 -N 9 -user_set_subdomains -Mdomains 1 -Ndomains 3 -overlap 1 -print_error -pc_gasm_print_subdomains > ex62.tmp 2>&1; \
	    if (${DIFF} output/ex62_2D_1.out ex62.tmp) then true; \
	    else printf "${PWD}\nPossible problem with ex62_2D_1, diffs above\n=========================================\n"; fi; \
	    ${RM} -f ex62.tmp


runex62_2D_2:
	-@${MPIEXEC} -n 2 ./ex62 -M 7 -N 9 -user_set_subdomains -Mdomains 1 -Ndomains 3 -overlap 1 -print_error -pc_gasm_print_subdomains > ex62.tmp 2>&1; \
	   if (${DIFF} output/ex62_2D_2.out ex62.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex62_2D_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex62.tmp

runex62_2D_3:
	-@${MPIEXEC} -n 3 ./ex62 -M 7 -N 9 -user_set_subdomains -Mdomains 1 -Ndomains 3 -overlap 1 -print_error -pc_gasm_print_subdomains > ex62.tmp 2>&1; \
	   if (${DIFF} output/ex62_2D_3.out ex62.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex62_2D_3, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex62.tmp

runex62_superlu_dist_1:
	-@${MPIEXEC} -n 1 ./ex62 -M 7 -N 9 -print_error -pc_gasm_total_subdomains 1 -pc_gasm_print_subdomains -sub_pc_type lu -sub_pc_factor_mat_solver_package superlu_dist > ex62.tmp 2>&1; \
	    if (${DIFF} output/ex62_superlu_dist_1.out ex62.tmp) then true; \
	    else printf "${PWD}\nPossible problem with ex62_superlu_dist_1, diffs above\n=========================================\n"; fi; \
	    ${RM} -f ex62.tmp

runex62_superlu_dist_2:
	-@${MPIEXEC} -n 2 ./ex62 -M 7 -N 9 -print_error -pc_gasm_total_subdomains 1 -pc_gasm_print_subdomains -sub_pc_type lu -sub_pc_factor_mat_solver_package superlu_dist > ex62.tmp 2>&1; \
	    if (${DIFF} output/ex62_superlu_dist_2.out ex62.tmp) then true; \
	    else printf "${PWD}\nPossible problem with ex62_superlu_dist_2, diffs above\n=========================================\n"; fi; \
	    ${RM} -f ex62.tmp

runex62_superlu_dist_3:
	-@${MPIEXEC} -n 3 ./ex62 -M 7 -N 9 -print_error -pc_gasm_total_subdomains 2 -pc_gasm_print_subdomains -sub_pc_type lu -sub_pc_factor_mat_solver_package superlu_dist > ex62.tmp 2>&1; \
	    if (${DIFF} output/ex62_superlu_dist_3.out ex62.tmp) then true; \
	    else printf "${PWD}\nPossible problem with ex62_superlu_dist_3, diffs above\n=========================================\n"; fi; \
	    ${RM} -f ex62.tmp

runex62_superlu_dist_4:
	-@${MPIEXEC} -n 4 ./ex62 -M 7 -N 9 -print_error -pc_gasm_total_subdomains 2 -pc_gasm_print_subdomains -sub_pc_type lu -sub_pc_factor_mat_solver_package superlu_dist > ex62.tmp 2>&1; \
	    if (${DIFF} output/ex62_superlu_dist_4.out ex62.tmp) then true; \
	    else printf "${PWD}\nPossible problem with ex62_superlu_dist_4, diffs above\n=========================================\n"; fi; \
	    ${RM} -f ex62.tmp

runex63:
	-@${MPIEXEC} -n 1 ./ex63 --filedir=${wPETSC_DIR}/share/petsc/datafiles/matrices/ --filename=amesos2_test_mat0.mtx --solver=SuperLU --print-residual=true -ksp_monitor -pc_type lu -pc_factor_mat_solver_package superlu -ksp_view -ksp_converged_reason  > ex63_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex63_1.out ex63_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with with ex63_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex63_1.tmp

runex63_2:
	-@${MPIEXEC} -n 1 ./ex63 --filedir=${wPETSC_DIR}/share/petsc/datafiles/matrices/ --filename=amesos2_test_mat0.mtx --solver=SuperLUDist --print-residual=true -ksp_monitor -pc_type lu -pc_factor_mat_solver_package superlu_dist -ksp_view -ksp_converged_reason  > ex63_2.tmp 2>&1;	  \
	   if (${DIFF} output/ex63_2.out ex63_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with with ex63_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex63_2.tmp

runex64:
	-@${MPIEXEC} -n 4 ./ex64  -ksp_monitor -pc_gasm_overlap  1  -sub_pc_type lu  -sub_pc_factor_mat_solver_package superlu_dist    -pc_gasm_total_subdomains 2 > ex64.tmp 2>&1;	  \
	   if (${DIFF} output/ex64.out ex64.tmp) then true; \
	   else printf "${PWD}\nPossible problem with with ex64, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex64.tmp

runex65:
	-@${MPIEXEC} -n 4 ./ex65  -ksp_monitor -pc_type mg -da_refine 3 > ex65.tmp 2>&1;	  \
	   if (${DIFF} output/ex65.out ex65.tmp) then true; \
	   else printf "${PWD}\nPossible problem with with ex65, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex65.tmp

runex66:
	-@${MPIEXEC} -n 1 ./ex66 -pc_type mg -pc_mg_type full  -ksp_monitor_short -da_refine 3  -mg_coarse_pc_type svd -ksp_view  > ex66.tmp 2>&1;         \
        ${DIFF} output/ex66_1.out ex66.tmp || printf "${PWD}\nPossible problem with ex66, diffs above\n=========================================\n"; \
        ${RM} -f ex66.tmp

runex66_2:
	-@${MPIEXEC} -n 4 ./ex66 -pc_type mg -pc_mg_type full  -ksp_monitor_short -da_refine 3  -mg_coarse_pc_type redundant -mg_coarse_redundant_pc_type svd -ksp_view > ex66_2.tmp 2>&1;   \
        ${DIFF} output/ex66_2.out ex66_2.tmp || printf "${PWD}\nPossible problem with ex66_2, diffs above\n=========================================\n"; \
        ${RM} -f ex66_2.tmp

runex67_symmetric_left:
	-@${MPIEXEC} -n 1 ./ex67 -ksp_view -ksp_converged_reason -pc_type sor -mat_no_inode -ksp_monitor_true_residual  -ksp_rtol 1.e-14 -ksp_max_it 12 -ksp_pc_side left  > ex67.tmp 2>&1;  \
        ${DIFF} output/ex67_symmetric_left.out ex67.tmp || printf "${PWD}\nPossible problem with ex67_symmetric_left, diffs above\n=========================================\n"; \
        ${RM} -f ex67.tmp

runex67_symmetric_right:
	-@${MPIEXEC} -n 1 ./ex67 -ksp_view -ksp_converged_reason -pc_type sor -mat_no_inode -ksp_monitor_true_residual  -ksp_rtol 1.e-14 -ksp_max_it 12 -ksp_pc_side right | sed 's/ATOL/RTOL/g' > ex67.tmp 2>&1;  \
        ${DIFF} output/ex67_symmetric_right.out ex67.tmp || printf "${PWD}\nPossible problem with ex67_symmetric_right, diffs above\n=========================================\n"; \
        ${RM} -f ex67.tmp

runex67_nonsymmetric_left:
	-@${MPIEXEC} -n 1 ./ex67 -symmetric false -ksp_view -ksp_converged_reason -pc_type jacobi -mat_no_inode -ksp_monitor_true_residual  -ksp_rtol 1.e-14 -ksp_max_it 12 -ksp_pc_side left | sed 's/ATOL/RTOL/g' > ex67.tmp 2>&1;  \
        ${DIFF} output/ex67_nonsymmetric_left.out ex67.tmp || printf "${PWD}\nPossible problem with ex67_nonsymmetric_left, diffs above\n=========================================\n"; \
        ${RM} -f ex67.tmp

runex67_nonsymmetric_right:
	-@${MPIEXEC} -n 1 ./ex67 -symmetric false -ksp_view -ksp_converged_reason -pc_type jacobi -mat_no_inode -ksp_monitor_true_residual  -ksp_rtol 1.e-14 -ksp_max_it 12 -ksp_pc_side right | sed 's/ATOL/RTOL/g'  > ex67.tmp 2>&1;  \
        ${DIFF} output/ex67_nonsymmetric_right.out ex67.tmp || printf "${PWD}\nPossible problem with ex67_nonsymmetric_right, diffs above\n=========================================\n"; \
        ${RM} -f ex67.tmp

TESTEXAMPLES_C		       = ex12.PETSc runex12 ex12.rm ex13.PETSc runex13 ex13.rm \
                                 ex15.PETSc runex15 ex15.rm ex16.PETSc runex16 ex16.rm \
                                 ex23.PETSc runex23 runex23_2 ex23.rm ex25.PETSc runex25_2 ex25.rm \
                                 ex18.PETSc runex18_3 ex18.rm \
                                 ex27.PETSc ex27.rm ex28.PETSc ex28.rm ex29.PETSc runex29_telescope ex29.rm \
                                 ex31.PETSc ex31.rm ex32.PETSc runex32 ex32.rm ex34.PETSc runex34 ex34.rm ex42.PETSc runex42 runex42_2 ex42.rm \
                                 ex43.PETSc runex43 runex43_2 runex43_bjacobi runex43_bjacobi_baij runex43_nested_gmg runex43_6 ex43.rm \
                                 ex45.PETSc runex45 runex45_2 runex45_telescope runex45_telescope_2 ex45.rm printdot \
                                 ex49.PETSc runex49 runex49_2 runex49_3 runex49_5 ex49.rm \
                                 ex51.PETSc runex51 ex51.rm \
                                 ex53.PETSc runex53 ex53.rm \
                                 ex54.PETSc runex54 runex54_Classical ex54.rm ex55.PETSc runex55 runex55_Classical runex55_NC ex55.rm\
                                 ex56.PETSc runex56 runex56_nns runex56_nns_telescope  ex56.rm \
                                 ex58.PETSc runex58 runex58_baij runex58_sbaij ex58.rm \
                                 ex60.PETSc runex60 runex60_2 runex60_3 ex60.rm \
                                 ex62.PETSc runex62_2D_1 runex62_2D_2 runex62_2D_3 ex62.rm ex65.PETSc runex65 ex65.rm
TESTEXAMPLES_C_NOTSINGLE       = ex18.PETSc runex18_bas ex18.rm ex25.PETSc runex25 ex25.rm ex43.PETSc runex43_3 ex43.rm ex67.PETSc printdot \
                                 runex67_symmetric_left runex67_symmetric_right runex67_nonsymmetric_left runex67_nonsymmetric_right ex67.rm
TESTEXAMPLES_C_NOCOMPLEX       = ex54.PETSc ex54.rm
TESTEXAMPLES_C_NOCOMPLEX_NOTSINGLE = ex23.PETSc runex23_3 ex23.rm  ex15.PETSc runex15_tsirm ex15.rm \
                                 ex34.PETSc runex34 runex34_2 ex34.rm \
                                 ex43.PETSc runex43_4 runex43_5 ex43.rm \
                                 ex49.PETSc runex49_6 runex49_7 runex49_8 ex49.rm \
                                 ex50.PETSc runex50 runex50_2 runex50_3 ex50.rm printdot \
                                 ex60.PETSc runex60_4 runex60_5 runex60_6 runex60_7 runex60_8 runex60_9 ex60.rm ex66.PETSc runex66 runex66_2 ex66.rm
TESTEXAMPLES_C_X	       = ex28.PETSc runex28 ex28.rm
TESTEXAMPLES_FORTRAN	       =                                   ex45f.PETSc runex45f ex45f.rm
TESTEXAMPLES_FORTRAN_NOTSINGLE  = ex14f.PETSc runex14f  ex14f.rm  ex22f.PETSc runex22f ex22f.rm ex21f.PETSc runex21f ex21f.rm  ex54f.PETSc runex54f ex54f.rm
TESTEXAMPLES_C_X_MPIUNI      =                                  ex23.PETSc runex23 ex23.rm
TESTEXAMPLES_C_COMPLEX	       = ex11.PETSc runex11 ex11.rm
TESTEXAMPLES_DATAFILESPATH     =                                  ex27.PETSc runex27 ex27.rm
TESTEXAMPLES_FORTRAN_NOCOMPLEX =
TESTEXAMPLES_FORTRAN_COMPLEX            = ex11f.PETSc runex11f ex11f.rm
TESTEXAMPLES_F90	                = ex13f90.PETSc runex13f90 ex13f90.rm ex44f.PETSc runex44f ex44f.rm 
#TESTEXAMPLES_F90_THREADSAFETY           = ex61f.PETSc runex61f ex61f.rm
TESTEXAMPLES_13		                = ex14f.PETSc ex14f.rm
TESTEXAMPLES_HYPRE	                = ex34.PETSc runex34_hyprestruct ex34.rm ex49.PETSc runex49_hypre_nullspace ex49.rm
TESTEXAMPLES_PARMETIS	                =
TESTEXAMPLES_MUMPS                      = ex53.PETSc runex53 runex53_2 ex53.rm \
                                          ex52.PETSc runex52 runex52_mumps runex52_mumps_2 runex52_mumps_3 ex52.rm ex52f.PETSc runex52f_mumps ex52f.rm
TESTEXAMPLES_SUPERLU                    = ex52.PETSc runex52_superlu_ilu runex52_superlu ex52.rm
TESTEXAMPLES_SUPERLU_DIST               =                                           ex52.PETSc runex52_superlu_dist ex52.rm ex64.PETSc runex64 ex64.rm \
	 			          ex62.PETSc runex62_superlu_dist_1 runex62_superlu_dist_2 runex62_superlu_dist_3 runex62_superlu_dist_4 ex62.rm
TESTEXAMPLES_STRUMPACK                  = ex52.PETSc runex52_strumpack_ilu runex52_strumpack runex52_strumpack_ilu_2 runex52_strumpack_2 ex52.rm
TESTEXAMPLES_VECCUDA                    =                                           ex46.PETSc runex46_aijcusparse ex46.rm
TESTEXAMPLES_CUSP                       =                                           ex46.PETSc runex46_aijcusp ex46.rm
TESTEXAMPLES_MOAB                       = ex35.PETSc runex35 ex35.rm ex36.PETSc runex36 ex36.rm
TESTEXAMPLES_MOAB_HDF5                  = ex35.PETSc runex35_2 runex35_3 ex35.rm ex36.PETSc runex36_2 ex36.rm
TESTEXAMPLES_TRILINOS                   = ex63.PETSc runex63 runex63_2 ex63.rm

include ${PETSC_DIR}/lib/petsc/conf/test
