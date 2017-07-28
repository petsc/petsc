
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/snes/examples/tests/
EXAMPLESC       = ex1.c ex3.c ex7.c ex17.c ex68.c ex69.c
EXAMPLESF       = ex1f.F90 ex12f.F ex18f90.F90
DIRS	        =
MANSEC          = SNES

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_SNES_LIB}
	${RM} ex1.o

ex1f: ex1f.o  chkopts
	-${FLINKER} -o ex1f ex1f.o ${PETSC_SNES_LIB}
	${RM} ex1f.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_SNES_LIB}
	${RM} ex2.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o ${PETSC_SNES_LIB}
	${RM} ex3.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o ${PETSC_SNES_LIB}
	${RM} ex4.o

ex5: ex5.o  chkopts
	-${CLINKER} -o ex5 ex5.o ${PETSC_SNES_LIB}
	${RM} ex5.o

ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o ${PETSC_SNES_LIB}
	${RM} ex6.o

ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o ${PETSC_SNES_LIB}
	${RM} ex7.o

ex9: ex9.o  chkopts
	-${CLINKER} -o ex9 ex9.o ${PETSC_SNES_LIB}
	${RM} ex9.o

ex10: ex10.o  chkopts
	-${FLINKER} -o ex10 ex10.o ${PETSC_SNES_LIB}
	${RM} ex10.o

ex11: ex11.o  chkopts
	-${CLINKER} -o ex11 ex11.o ${PETSC_SNES_LIB}
	${RM} ex11.o

ex12f: ex12f.o  chkopts
	-${FLINKER} -o ex12f ex12f.o ${PETSC_SNES_LIB}
	${RM} ex12f.o

ex13: ex13.o  chkopts
	-${CLINKER} -o ex13 ex13.o ${PETSC_SNES_LIB}
	${RM} ex13.o

ex14f: ex14f.o  chkopts
	-${FLINKER} -o ex14f ex14f.o ${PETSC_SNES_LIB}
	${RM} ex14f.o

ex17: ex17.o chkopts
	-${CLINKER} -o ex17 ex17.o ${PETSC_SNES_LIB}
	${RM} ex17.o

ex18f90: ex18f90.o chkopts
	-${FLINKER} -o ex18f90 ex18f90.o ${PETSC_SNES_LIB}
	${RM} ex18f90.o

ex68: ex68.o chkopts
	-${CLINKER} -o ex68 ex68.o ${PETSC_SNES_LIB}
	${RM} ex68.o

ex69: ex69.o chkopts
	-${CLINKER} -o ex69 ex69.o ${PETSC_SNES_LIB}
	${RM} ex69.o
#--------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 -ksp_gmres_cgs_refinement_type refine_always > ex1_1.tmp 2>&1;	 \
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1_1.tmp
runex1_2:
	-@${MPIEXEC} -n 1 ./ex1 -snes_monitor_short -snes_type newtontr -ksp_gmres_cgs_refinement_type refine_always > ex1_2.tmp 2>&1;\
	   if (${DIFF} output/ex1_2.out ex1_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1_2.tmp
runex1_3:
	-@${MPIEXEC} -n 1 ./ex1 -snes_monitor_short -mat_coloring_type sl  -snes_fd_coloring -mx 8 -my 11 -ksp_gmres_cgs_refinement_type refine_always > ex1_3.tmp 2>&1;\
	   if (${DIFF} output/ex1_3.out ex1_3.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1_3, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1_3.tmp
runex1f:
	-@${MPIEXEC} -n 1 ./ex1f -snes_monitor_short -nox -snes_type newtontr -ksp_gmres_cgs_refinement_type refine_always > ex1f_1.tmp 2>&1;\
	   if (${DIFF} output/ex1f_1.out ex1f_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1f_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1f_1.tmp
runex1f_2:
	-@${MPIEXEC} -n 1 ./ex1f -snes_monitor_short -nox -snes_fd -ksp_gmres_cgs_refinement_type refine_always > ex1f_2.tmp 2>&1;\
	   if (${DIFF} output/ex1f_2.out ex1f_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1f_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1f_2.tmp
runex1f_3:
	-@${MPIEXEC} -n 1 ./ex1f -snes_monitor_short -nox -snes_fd_coloring -mat_coloring_type sl -ksp_gmres_cgs_refinement_type refine_always | sort -b > ex1f_3.tmp 2>&1;\
           if (sort -b output/ex1f_3.out | ${DIFF} - ex1f_3.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1f_3, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1f_3.tmp
runex3:
	-@${MPIEXEC} -n 1 ./ex3 -snes_converged_reason  > ex3_1.tmp 2>&1;	 \
	   if (${DIFF} output/ex3_1.out ex3_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex3_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex3_1.tmp
runex4:
	-@${MPIEXEC} -n 1 ./ex4 -snes_converged_reason -snes_linesearch_monitor -snes_linesearch_type l2 | grep Inf > ex4_1.tmp 2>&1;	 \
	   if (${DIFF} output/ex4_1.out ex4_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex4_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex4_1.tmp
runex7:
	-@${MPIEXEC} -n 1 ./ex7 -ksp_gmres_cgs_refinement_type refine_always -snes_monitor_cancel -snes_monitor_short > ex7_1.tmp 2>&1; \
	   if (${DIFF} output/ex7_1.out ex7_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex7_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex7_1.tmp
runex7_2:
	-@${MPIEXEC} -n 1 ./ex7 -variant -ksp_gmres_cgs_refinement_type refine_always -snes_monitor_cancel -snes_monitor_short > ex7_2.tmp 2>&1; \
	   if (${DIFF} output/ex7_1.out ex7_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex7_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex7_2.tmp
runex12f:
	-@${MPIEXEC} -n 2 ./ex12f -ksp_gmres_cgs_refinement_type refine_always -n 10 -snes_monitor_short  > ex12_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex12_1.out ex12_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex12f_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex12_1.tmp usermodule.mod
runex17:
	-@${MPIEXEC} -n 1 ./ex17 -snes_monitor_short > ex17_1.tmp 2>&1; \
	   ${DIFF} output/ex17_1.out ex17_1.tmp || printf "${PWD}\nPossible problem with ex17, diffs above\n=========================================\n"; \
	   ${RM} -f ex17_1.tmp

runex18f90:
	-@${MPIEXEC} -n 1 ./ex18f90 > ex18f90_1.tmp 2>&1; \
	   ${DIFF} output/ex18f90_1.out ex18f90_1.tmp || printf "${PWD}\nPossible problem with ex18f90, diffs above\n=========================================\n"; \
	   ${RM} -f ex18f90_1.tmp base_module.mod extended_module.mod function_module.mod

runex69:
	-@${MPIEXEC} -n 1 ./ex69 -snes_converged_reason -ksp_converged_reason > ex69_1.tmp 2>&1; \
	   ${DIFF} output/ex69_1.out ex69_1.tmp || printf "${PWD}\nPossible problem with ex69, diffs above\n=========================================\n"; \
	   ${RM} -f ex69_1.tmp

runex69_2:
	-@${MPIEXEC} -n 1 ./ex69 -snes_converged_reason -ksp_converged_reason -error_in_matmult > ex69_2.tmp 2>&1; \
	   ${DIFF} output/ex69_2.out ex69_2.tmp || printf "${PWD}\nPossible problem with ex69_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex69_2.tmp

runex69_3:
	-@${MPIEXEC} -n 1 ./ex69 -snes_converged_reason -ksp_converged_reason -error_in_pcapply > ex69_3.tmp 2>&1; \
	   ${DIFF} output/ex69_3.out ex69_3.tmp || printf "${PWD}\nPossible problem with ex69_3, diffs above\n=========================================\n"; \
	   ${RM} -f ex69_3.tmp

runex69_4:
	-@${MPIEXEC} -n 1 ./ex69 -snes_converged_reason -ksp_converged_reason -error_in_pcsetup > ex69_4.tmp 2>&1; \
	   ${DIFF} output/ex69_4.out ex69_4.tmp || printf "${PWD}\nPossible problem with ex69_4, diffs above\n=========================================\n"; \
	   ${RM} -f ex69_4.tmp

runex69_5:
	-@${MPIEXEC} -n 1 ./ex69 -snes_converged_reason -ksp_converged_reason -error_in_pcsetup -pc_type bjacobi > ex69_5.tmp 2>&1; \
	   ${DIFF} output/ex69_5.out ex69_5.tmp || printf "${PWD}\nPossible problem with ex69_5, diffs above\n=========================================\n"; \
	   ${RM} -f ex69_5.tmp
runex69_5_fieldsplit:
	-@${MPIEXEC} -n 1 ./ex69 -snes_converged_reason -ksp_converged_reason -error_in_pcsetup -pc_type fieldsplit > ex69_5.tmp 2>&1; \
	   ${DIFF} output/ex69_5.out ex69_5.tmp || printf "${PWD}\nPossible problem with ex69_5_fieldsplit, diffs above\n=========================================\n"; \
	   ${RM} -f ex69_5.tmp

runex69_6:
	-@${MPIEXEC} -n 1 ./ex69 -snes_converged_reason -ksp_converged_reason -error_in_domainmf -snes_mf > ex69_6.tmp 2>&1; \
	   ${DIFF} output/ex69_6.out ex69_6.tmp || printf "${PWD}\nPossible problem with ex69_6, diffs above\n=========================================\n"; \
	   ${RM} -f ex69_6.tmp

runex69_7:
	-@${MPIEXEC} -n 1 ./ex69 -snes_converged_reason -ksp_converged_reason -error_in_domain  > ex69_7.tmp 2>&1; \
	   ${DIFF} output/ex69_7.out ex69_7.tmp || printf "${PWD}\nPossible problem with ex69_7, diffs above\n=========================================\n"; \
	   ${RM} -f ex69_7.tmp

runex69_8:
	-@${MPIEXEC} -n 1 ./ex69 -snes_converged_reason -ksp_converged_reason -error_in_domain -snes_mf > ex69_8.tmp 2>&1; \
	   ${DIFF} output/ex69_8.out ex69_8.tmp || printf "${PWD}\nPossible problem with ex69_8, diffs above\n=========================================\n"; \
	   ${RM} -f ex69_8.tmp

TESTEXAMPLES_C		       = ex1.PETSc  runex1_3 ex1.rm ex3.PETSc runex3 ex3.rm  ex68.PETSc ex68.rm \
                                 ex69.PETSc runex69  runex69_2 runex69_3 runex69_4 runex69_5 runex69_5_fieldsplit runex69_6 runex69_7 ex69.rm
TESTEXAMPLES_C_NOTSINGLE       = ex1.PETSc runex1 runex1_2 ex1.rm ex17.PETSc runex17 ex17.rm
TESTEXAMPLES_C_X	       = ex7.PETSc runex7 runex7_2 ex7.rm
TESTEXAMPLES_FORTRAN	       = ex12f.PETSc runex12f ex12f.rm
TESTEXAMPLES_FORTRAN_NOTSINGLE = ex1f.PETSc runex1f_2 runex1f_3 ex1f.rm
TESTEXAMPLES_C_X_MPIUNI        = ex7.PETSc ex7.rm ex1.PETSc runex1 runex1_2 runex1_3 ex1.rm
TESTEXAMPLES_FORTRAN_NOCOMPLEX =
TESTEXAMPLES_FORTRAN_MPIUNI    = ex1f.PETSc runex1f ex1f.rm
TESTEXAMPLES_F2003             = ex18f90.PETSc runex18f90 ex18f90.rm


include ${PETSC_DIR}/lib/petsc/conf/test
