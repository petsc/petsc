
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =	 
LOCDIR          = src/snes/examples/tests/
EXAMPLESC       = ex1.c ex4.c ex5.c ex7.c ex9.c ex11.c ex13.c
EXAMPLESF       = ex1f.F ex12f.F
DIRS	        = 
MANSEC          = SNES

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

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

ex8: ex8.o  chkopts
	-${CLINKER} -o ex8 ex8.o ${PETSC_SNES_LIB}
	${RM} ex8.o

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
#--------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 -ksp_gmres_cgs_refinement_type refine_always > ex1_1.tmp 2>&1;	 \
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   ${RM} -f ex1_1.tmp
runex1_2:
	-@${MPIEXEC} -n 1 ./ex1 -snes_monitor_short -snes_type tr -ksp_gmres_cgs_refinement_type refine_always > ex1_2.tmp 2>&1;\
	   if (${DIFF} output/ex1_2.out ex1_2.tmp) then true; \
	   else echo "Possible problem with ex1_2, diffs above"; fi; \
	   ${RM} -f ex1_2.tmp
runex1_3:
	-@${MPIEXEC} -n 1 ./ex1 -snes_monitor_short -mat_coloring_type sl  -snes_fd_coloring -mx 8 -my 11 \
                -ksp_gmres_cgs_refinement_type refine_always > ex1_3.tmp 2>&1;\
	   if (${DIFF} output/ex1_3.out ex1_3.tmp) then true; \
	   else echo "Possible problem with ex1_3, diffs above"; fi; \
	   ${RM} -f ex1_3.tmp
runex1f:
	-@${MPIEXEC} -n 1 ./ex1f -snes_monitor_short -nox -snes_type tr -ksp_gmres_cgs_refinement_type refine_always > ex1f_1.tmp 2>&1;\
	   if (${DIFF} output/ex1f_1.out ex1f_1.tmp) then true; \
	   else echo "Possible problem with ex1f_1, diffs above"; fi; \
	   ${RM} -f ex1f_1.tmp
runex1f_2:
	-@${MPIEXEC} -n 1 ./ex1f -snes_monitor_short -nox -snes_fd -ksp_gmres_cgs_refinement_type refine_always > ex1f_2.tmp 2>&1;\
	   if (${DIFF} output/ex1f_2.out ex1f_2.tmp) then true; \
	   else echo "Possible problem with ex1f_2, diffs above"; fi; \
	   ${RM} -f ex1f_2.tmp
runex1f_3:
	-@${MPIEXEC} -n 1 ./ex1f -snes_monitor_short -nox -snes_fd_coloring -ksp_gmres_cgs_refinement_type refine_always > ex1f_3.tmp 2>&1;\
	   if (${DIFF} output/ex1f_2.out ex1f_3.tmp) then true; \
	   else echo "Possible problem with ex1f_3, diffs above"; fi; \
	   ${RM} -f ex1f_3.tmp
runex2:
	-@${MPIEXEC} -n 1 ./ex2	-ksp_gmres_cgs_refinement_type refine_always > ex2_1.tmp 2>&1;   \
	   if (${DIFF} output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi; \
	   ${RM} -f ex2_1.tmp
runex3:
	-@${MPIEXEC} -n 1 ./ex3 -ksp_gmres_cgs_refinement_type refine_always -nox -snes_monitor_short -snes_view \
		-pc_type jacobi >ex3_1.tmp 2>&1;	\
	   if (${DIFF} output/ex3_1.out ex3_1.tmp) then true; \
	   else echo "Possible problem with ex3_1, diffs above"; fi; \
	   ${RM} -f ex3_1.tmp
runex3_2:
	-@${MPIEXEC} -n 1 ./ex3 -ksp_gmres_cgs_refinement_type refine_always -nox -snes_monitor_short -snes_type tr -snes_view \
		-pc_type jacobi > ex3_2.tmp 2>&1; \
	   if (${DIFF} output/ex3_2.out ex3_2.tmp) then true; \
	   else echo "Possible problem with ex3_2, diffs above"; fi; \
	   ${RM} -f ex3_2.tmp
runex4:
	-@${MPIEXEC} -n 1 ./ex4 -ksp_gmres_cgs_refinement_type refine_always -nox > ex4_1.tmp 2>&1;	 \
	   if (${DIFF} output/ex4_1.out ex4_1.tmp) then true; \
	   else echo "Possible problem with ex4_1, diffs above"; fi; \
	   ${RM} -f ex4_1.tmp
runex4_2:
	-@${MPIEXEC} -n 1 ./ex4 -ksp_gmres_cgs_refinement_type refine_always -snes_monitor_short -nox -snes_type tr > ex4_2.tmp 2>&1;\
	   if (${DIFF} output/ex4_2.out ex4_2.tmp) then true; \
	   else echo "Possible problem with ex4_2, diffs above"; fi; \
	   ${RM} -f ex4_2.tmp
runex5:
	-@${MPIEXEC} -n 1 ./ex5 -ksp_gmres_cgs_refinement_type refine_always -snes_monitor_short > ex5_1.tmp 2>&1;   \
	   if (${DIFF} output/ex5_1.out ex5_1.tmp) then true; \
	   else echo "Possible problem with ex5_1, diffs above"; fi; \
	   ${RM} -f ex5_1.tmp
runex5_2:
	-@${MPIEXEC} -n 1 ./ex5 -ksp_gmres_cgs_refinement_type refine_always -snes_mf -snes_monitor_short > ex5_2.tmp 2>&1;   \
	   if (${DIFF} output/ex5_2.out ex5_2.tmp) then true; \
	   else echo "Possible problem with ex5_2, diffs above"; fi; \
	   ${RM} -f ex5_2.tmp
runex6:
	-@${MPIEXEC} -n 4 ./ex6 -ksp_gmres_cgs_refinement_type refine_always -snes_mf -Nx 4 -Ny 1 -snes_monitor_short > ex6_1.tmp 2>&1; \
	   if (${DIFF} output/ex6_1.out ex6_1.tmp) then true; \
	   else echo "Possible problem with ex6_1, diffs above"; fi; \
	   ${RM} -f ex6_1.tmp
runex6_2:
	-@${MPIEXEC} -n 4 ./ex6 -ksp_gmres_cgs_refinement_type refine_always -mat_type mpiaij -Nx 2 -Ny 2 -snes_monitor_short > ex6_2.tmp 2>&1; \
	   if (${DIFF} output/ex6_2.out ex6_2.tmp) then true; \
	   else echo "Possible problem with ex6_2, diffs above"; fi; \
	   ${RM} -f ex6_2.tmp
runex6_3:
	-@${MPIEXEC} -n 4 ./ex6 -ksp_gmres_cgs_refinement_type refine_always -snes_mf -Nx 2 -Ny 2 -star -snes_monitor_short>ex6_3.tmp 2>&1;\
	   if (${DIFF} output/ex6_1.out ex6_3.tmp) then true; \
	   else echo "Possible problem with ex6_3, diffs above"; fi; \
	   ${RM} -f ex6_3.tmp
runex6_4:
	-@${MPIEXEC} -n 4 ./ex6 -ksp_gmres_cgs_refinement_type refine_always -mat_type mpiaij -Nx 2 -Ny 2 -snes_monitor_short -pc_type asm \
	   -pc_asm_blocks 4 -pc_asm_overlap 1 > ex6_4.tmp 2>&1; \
	   if (${DIFF} output/ex6_4.out ex6_4.tmp) then true; \
	   else echo "Possible problem with ex6_4, diffs above"; fi; \
	   ${RM} -f ex6_4.tmp
runex6_5:
	-@${MPIEXEC} -n 4 ./ex6 -ksp_gmres_cgs_refinement_type refine_always -mat_type mpiaij -Nx 2 -Ny 2 -snes_monitor_short -pc_type bjacobi \
	   -snes_view > ex6_5.tmp 2>&1; \
	   if (${DIFF} output/ex6_5.out ex6_5.tmp) then true; \
	   else echo "Possible problem with ex6_5, diffs above"; fi; \
	   ${RM} -f ex6_5.tmp
runex6_6:
	-@${MPIEXEC} -n 4 ./ex6 -ksp_gmres_cgs_refinement_type refine_always -snes_fd -Nx 2 -Ny 2 -snes_monitor_short > ex6_6.tmp 2>&1; \
	   if (${DIFF} output/ex6_6.out ex6_6.tmp) then true; \
	   else echo "Possible problem with ex6_6, diffs above"; fi; \
	   ${RM} -f ex6_6.tmp
runex6_7:
	-@${MPIEXEC} -n 3 ./ex6 -ksp_gmres_cgs_refinement_type refine_always -snes_fd  -snes_monitor_short > ex6_7.tmp 2>&1;\
	   if (${DIFF} output/ex6_7.out ex6_7.tmp) then true; \
	   else echo "Possible problem with ex6_7, diffs above"; fi; \
	   ${RM} -f ex6_7.tmp
runex7:
	-@${MPIEXEC} -n 1 ./ex7 -ksp_gmres_cgs_refinement_type refine_always -snes_monitor_cancel -snes_monitor_short > ex7_1.tmp 2>&1; \
	   if (${DIFF} output/ex7_1.out ex7_1.tmp) then true; \
	   else echo "Possible problem with ex7_1, diffs above"; fi; \
	   ${RM} -f ex7_1.tmp
runex7_2:
	-@${MPIEXEC} -n 1 ./ex7 -variant -ksp_gmres_cgs_refinement_type refine_always -snes_monitor_cancel -snes_monitor_short > ex7_2.tmp 2>&1; \
	   if (${DIFF} output/ex7_1.out ex7_2.tmp) then true; \
	   else echo "Possible problem with ex7_2, diffs above"; fi; \
	   ${RM} -f ex7_2.tmp
runex8:
	-@${MPIEXEC} -n 2 ./ex8 -ksp_gmres_cgs_refinement_type refine_always -snes_monitor_short  > ex8_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex8_1.out ex8_1.tmp) then true; \
	   else echo "Possible problem with ex8_1, diffs above"; fi; \
	   ${RM} -f ex8_1.tmp
runex8_2:
	-@${MPIEXEC} -n 3 ./ex8 -ksp_gmres_cgs_refinement_type refine_always -pc_type asm  -mat_type mpiaij  \
	   -snes_monitor_short > ex8_2.tmp 2>&1; \
	   if (${DIFF} output/ex8_2.out ex8_2.tmp) then true; \
	   else echo "Possible problem with ex8_2, diffs above"; fi; \
	   ${RM} -f ex8_2.tmp

runex9:
	-@${MPIEXEC} -n 2 ./ex9 -ksp_gmres_cgs_refinement_type refine_always -snes_monitor_short  > ex9_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex9_1.out ex9_1.tmp) then true; \
	   else echo "Possible problem with ex9_1, diffs above"; fi; \
	   ${RM} -f ex9_1.tmp

runex9_2:
	-@${MPIEXEC} -n 8 ./ex9 -ksp_gmres_cgs_refinement_type refine_always -snes_monitor_short  > ex9_2.tmp 2>&1;	  \
	   if (${DIFF} output/ex9_1.out ex9_2.tmp) then true; \
	   else echo "Possible problem with ex9_1, diffs above"; fi; \
	   ${RM} -f ex9_2.tmp
runex9_3:
	-@${MPIEXEC} -n 8 ./ex9 -ksp_gmres_cgs_refinement_type refine_always -star -snes_monitor_short  > ex9_3.tmp 2>&1;	\
	   if (${DIFF} output/ex9_1.out ex9_3.tmp) then true; \
	   else echo "Possible problem with ex9_1, diffs above"; fi; \
	   ${RM} -f ex9_3.tmp
runex10:
	-@${MPIEXEC} -n 2 ./ex10 -ksp_gmres_cgs_refinement_type refine_always -n 10 -snes_monitor_short  > ex10_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex10_1.out ex10_1.tmp) then true; \
	   else echo "Possible problem with ex10_1, diffs above"; fi; \
	   ${RM} -f ex10_1.tmp
runex12f:
	-@${MPIEXEC} -n 2 ./ex12f -ksp_gmres_cgs_refinement_type refine_always -n 10 -snes_monitor_short  > ex12_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex12_1.out ex12_1.tmp) then true; \
	   else echo "Possible problem with ex12_1, diffs above"; fi; \
	   ${RM} -f ex12_1.tmp
runex14f:
	-@${MPIEXEC} -n 1 ./ex14f -snes_monitor_short > ex14_1.tmp 2>&1; \
	   if (${DIFF} output/ex14_1.out ex14_1.tmp) then true; \
	   else echo "Possible problem with ex14f_1, diffs above"; fi; \
	   ${RM} -f ex14_1.tmp

#
#  runex9_2 and runex9_3 are not included in RUNEXAMPLES_2 because 
# they require too many processors
#
TESTEXAMPLES_C		       = ex1.PETSc runex1 runex1_2 runex1_3 ex1.rm ex11.PETSc ex11.rm
TESTEXAMPLES_C_X11	       = ex7.PETSc runex7 runex7_2 ex7.rm ex9.PETSc runex9 ex9.rm ex13.PETSc ex13.rm
TESTEXAMPLES_FORTRAN	       = ex12f.PETSc runex12f ex12f.rm  ex1f.PETSc runex1f_2 runex1f_3 ex1f.rm
TESTEXAMPLES_C_X11_MPIUNI      = ex7.PETSc ex7.rm ex9.PETSc ex9.rm ex1.PETSc runex1 runex1_2 runex1_3 ex1.rm 
TESTEXAMPLES_FORTRAN_NOCOMPLEX = 
TESTEXAMPLES_FORTRAN_MPIUNI    = ex1f.PETSc runex1f ex1f.rm


include ${PETSC_DIR}/conf/test
