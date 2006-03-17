
CFLAGS	         = -I${PETSC_DIR}/src/dm/mesh/sieve ${BOOST_INCLUDE} ${TRIANGLE_INCLUDE} ${TETGEN_INCLUDE}
FFLAGS	         = 
CPPFLAGS         = -Wno-shadow
FPPFLAGS         =
LOCDIR           = src/ksp/ksp/examples/tutorials/
EXAMPLESC        = ex1.c ex2.c ex3.c ex4.c ex5.c ex7.c ex8.c ex9.c ex10.c ex11.c \
                   ex12.c ex13.c ex15.c ex16.c ex22.c ex23.c ex25.c ex27.c ex28.c ex29.c ex32.c ex33.c ex34.c
EXAMPLESF        = ex1f.F ex2f.F ex6f.F ex11f.F ex13f90.F ex14f.F ex15f.F ex21f.F ex22f.F
MANSEC           = KSP
CLEANFILES       = rhs.vtk solution.vtk
NP               = 1
DISPLAY          = :0.0

include ${PETSC_DIR}/bmake/common/base

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

ex35: ex35.o chkopts
	-${CLINKER} -o ex35 ex35.o ${PETSC_SNES_LIB}
	${RM} ex35.o

ex34: ex34.o chkopts
	-${CLINKER} -o ex34 ex34.o ${PETSC_SNES_LIB}
	${RM} ex34.o

#----------------------------------------------------------------------------
runex1:
	-@${MPIRUN} -np 1 ./ex1 -ksp_smonitor -ksp_gmres_cgs_refinement_type refine_always > ex1_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   ${RM} -f ex1_1.tmp
runex1_2:
	-@${MPIRUN} -np 1 ./ex1 -pc_type sor -pc_sor_symmetric -ksp_smonitor -ksp_gmres_cgs_refinement_type refine_always >\
	   ex1_2.tmp 2>&1;   \
	   if (${DIFF} output/ex1_2.out ex1_2.tmp) then true; \
	   else echo "Possible problem with ex1_2, diffs above"; fi; \
	   ${RM} -f ex1_2.tmp
runex1_3:
	-@${MPIRUN} -np 1 ./ex1 -pc_type eisenstat -ksp_smonitor -ksp_gmres_cgs_refinement_type refine_always >\
	   ex1_3.tmp 2>&1;   \
	   if (${DIFF} output/ex1_3.out ex1_3.tmp) then true; \
	   else echo "Possible problem with ex1_3, diffs above"; fi; \
	   ${RM} -f ex1_3.tmp
runex1f:
	-@${MPIRUN} -np 1 ./ex1f -ksp_smonitor -ksp_gmres_cgs_refinement_type refine_always > ex1f_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex1f_1.out ex1f_1.tmp) then true; \
	   else echo "Possible problem with ex1f_1, diffs above"; fi; \
	   ${RM} -f ex1f_1.tmp
runex2:
	-@${MPIRUN} -np 1 ./ex2 -ksp_smonitor -m 5 -n 5 -ksp_gmres_cgs_refinement_type refine_always > ex2_1.tmp 2>&1; \
	   if (${DIFF} output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi; \
	   ${RM} -f ex2_1.tmp
runex2_2:
	-@${MPIRUN} -np 2 ./ex2 -ksp_smonitor -m 5 -n 5 -ksp_gmres_cgs_refinement_type refine_always > ex2_2.tmp 2>&1; \
	   if (${DIFF} output/ex2_2.out ex2_2.tmp) then true; \
	   else echo "Possible problem with ex2_2, diffs above"; fi; \
	   ${RM} -f ex2_2.tmp
runex2_3:
	-@${MPIRUN} -np 1 ./ex2 -pc_type sor -pc_sor_symmetric -ksp_smonitor -ksp_gmres_cgs_refinement_type refine_always > \
	    ex2_3.tmp 2>&1;   \
	   if (${DIFF} output/ex2_3.out ex2_3.tmp) then true; \
	   else echo "Possible problem with ex2_3, diffs above"; fi; \
	   ${RM} -f ex2_3.tmp
runex2_4:
	-@${MPIRUN} -np 1 ./ex2 -pc_type eisenstat -ksp_smonitor -ksp_gmres_cgs_refinement_type refine_always >\
	    ex2_4.tmp 2>&1;   \
	   if (${DIFF} output/ex2_4.out ex2_4.tmp) then true; \
	   else echo "Possible problem with ex2_4, diffs above"; fi; \
	   ${RM} -f ex2_4.tmp
runex2_5:
	-@${MPIRUN} -np 2 ./ex2 -ksp_smonitor -m 5 -n 5 -mat_view_draw -ksp_gmres_cgs_refinement_type refine_always    > ex2_5.tmp 2>&1; \
	   if (${DIFF} output/ex2_2.out ex2_5.tmp) then true; \
	   else echo "Possible problem with ex2_5, diffs above"; fi; \
	   ${RM} -f ex2_5.tmp
runex2_6:
	-@${MPIRUN} -np 2 ./ex2 -pc_type none -vec_type esi -vec_esi_type esi::petsc::Vector \
               -mat_type esi -mat_esi_type esi::petsc::Matrix -is_esi_type esi::petsc::IndexSpace   > ex2_6.tmp 2>&1; \
	   if (${DIFF} output/ex2_6.out ex2_6.tmp) then true; \
	   else echo "Possible problem with ex2_6, diffs above"; fi; \
	   ${RM} -f ex2_6.tmp
runex2f:
	-@${MPIRUN} -np 2 ./ex2f -pc_type jacobi -ksp_smonitor -ksp_gmres_cgs_refinement_type refine_always > ex2f_1.tmp 2>&1; \
	   if (${DIFF} output/ex2f_1.out ex2f_1.tmp) then true; \
	   else echo "Possible problem with ex2f_1, diffs above"; fi; \
	   ${RM} -f ex2f_1.tmp
runex4:
	-@${MPIRUN} -np 3  ./ex4 -ksp_smonitor -ksp_gmres_cgs_refinement_type refine_always > ex4_1.tmp 2>&1; \
	   if (${DIFF} output/ex4_1.out ex4_1.tmp) then true; \
	   else echo "Possible problem with ex4_1, diffs above"; fi; \
	   ${RM} -f ex4_1.tmp
runex5:
	-@${MPIRUN} -np 1 ./ex5 -pc_type jacobi -ksp_smonitor -ksp_gmres_cgs_refinement_type refine_always > ex5_1.tmp 2>&1; \
	   if (${DIFF} output/ex5_1.out ex5_1.tmp) then true; \
	   else echo "Possible problem with ex5_1, diffs above"; fi; \
	   ${RM} -f ex5_1.tmp
runex5_2:
	-@${MPIRUN} -np 2 ./ex5 -pc_type jacobi -ksp_smonitor -ksp_gmres_cgs_refinement_type refine_always \
	   -ksp_rtol .000001 > ex5_2.tmp 2>&1;   \
	   if (${DIFF} output/ex5_2.out ex5_2.tmp) then true; \
	   else echo "Possible problem with ex5_2, diffs above"; fi; \
	   ${RM} -f ex5_2.tmp

# Note: We test these BlockSolve95 examples differently from the usual
# because the coloring heuristic is machine-dependent; convergence rates
# may differ slightly on various architectures.
runex5_3:
	-@${MPIRUN} -np 2 ./ex5 -pc_type icc -ksp_type cg -mat_type mpirowbs -ksp_rtol .000001;\
	echo "ex5_3: This program tested the ICC preconditioner in BlockSolve95."
runex5_4:
	-@${MPIRUN} -np 2 ./ex5 -pc_type ilu -mat_nonsym -mat_type mpirowbs -ksp_gmres_cgs_refinement_type refine_always\
	   -ksp_rtol .000001;\
	echo "ex5_4: This program tested the ILU preconditioner in BlockSolve95."
runex5_5:
	-@${MPIRUN} -np 2 ./ex5 -mat_type mpibdiag -mat_block_size 2 -ksp_gmres_cgs_refinement_type refine_always > ex5_5.tmp 2>&1; \
	   if (${DIFF} output/ex5_5.out ex5_5.tmp) then true; \
	   else echo "Possible problem with ex5_5, diffs above"; fi; \
	   ${RM} -f ex5_5.tmp
runex6f:
	-@${MPIRUN} -np 1 ./ex6f -pc_type jacobi -mat_view -ksp_smonitor -ksp_gmres_cgs_refinement_type refine_always > ex6f_1.tmp 2>&1; \
	   if (${DIFF} output/ex6f_1.out ex6f_1.tmp) then true; \
	   else echo "Possible problem with ex6f_1, diffs above"; fi; \
	   ${RM} -f ex6f_1.tmp
runex7:
	-@${MPIRUN} -np 2 ./ex7 -ksp_smonitor -nokspview -ksp_gmres_cgs_refinement_type refine_always> ex7_1.tmp 2>&1; \
	   if (${DIFF} output/ex7_1.out ex7_1.tmp) then true; \
	   else echo "Possible problem with ex7_1, diffs above"; fi; \
	   ${RM} -f ex7_1.tmp
runex9:
	-@${MPIRUN} -np 1 ./ex9 -t 2 -pc_type jacobi -ksp_smonitor -ksp_type gmres -ksp_gmres_cgs_refinement_type refine_always \
	   -s2_ksp_type bcgs -s2_pc_type jacobi -s2_ksp_smonitor \
           > ex9_1.tmp 2>&1; \
	   if (${DIFF} output/ex9_1.out ex9_1.tmp) then true; \
	   else echo "Possible problem with ex9_1, diffs above"; fi; \
	   ${RM} -f ex9_1.tmp
runex10:
	-@${MPIRUN} -np 2 ./ex10 -f0 ../../../../mat/examples/matbinary.cplx.ex -ksp_gmres_cgs_refinement_type refine_always -options_left off > ex10_1.tmp 2>&1; \
	   if (${DIFF} output/ex10_1.out ex10_1.tmp) then true; \
	   else echo "Possible problem with ex10_1, diffs above"; fi; \
	   ${RM} -f ex10_1.tmp
runex10_2:
	-@${MPIRUN} -np 2 ./ex10 -options_left no -ksp_type bicg \
	   -f0 ${DATAFILESPATH}/matrices/medium > ex10_2.tmp 2>&1; \
	   if (${DIFF} output/ex10_2.out ex10_2.tmp) then true; \
	   else echo "Possible problem with ex10_2, diffs above"; fi; \
	   ${RM} -f ex10_2.tmp
runex10_3:
	-@${MPIRUN} -np 2 ./ex10 -options_left no -ksp_type bicg -pc_type asm \
	   -f0 ${DATAFILESPATH}/matrices/medium > ex10_3.tmp 2>&1; \
	   if (${DIFF} output/ex10_3.out ex10_3.tmp) then true; \
	   else echo "Possible problem with ex10_3, diffs above"; fi; \
	   ${RM} -f ex10_3.tmp
runex10_4:
	-@${MPIRUN} -np 1 ./ex10 -options_left no -ksp_type bicg -pc_type lu \
	   -f0 ${DATAFILESPATH}/matrices/medium > ex10_4.tmp 2>&1; \
	   if (${DIFF} output/ex10_4.out ex10_4.tmp) then true; \
	   else echo "Possible problem with ex10_4, diffs above"; fi; \
	   ${RM} -f ex10_4.tmp
runex10_5:
	-@${MPIRUN} -np 1 ./ex10 -options_left no -ksp_type bicg \
	   -f0 ${DATAFILESPATH}/matrices/medium > ex10_5.tmp 2>&1; \
	   if (${DIFF} output/ex10_5.out ex10_5.tmp) then true; \
	   else echo "Possible problem with ex10_5, diffs above"; fi; \
	   ${RM} -f ex10_5.tmp
runex10_6:
	-@${MPIRUN} -np 1 ./ex10 -pc_factor_levels 2 -pc_factor_fill 1.73 -ksp_gmres_cgs_refinement_type refine_always \
	   -f0 ${DATAFILESPATH}/matrices/fem1 > ex10_6.tmp 2>&1; \
	   if (${DIFF} output/ex10_6.out ex10_6.tmp) then true; \
	   else echo "Possible problem with ex10_6, diffs above"; fi; \
	   ${RM} -f ex10_6.tmp
BS = 2 3 4 5 6 7 8
runex10_7:
	-@touch ex10_7.tmp
	-@for bs in ${BS}; do \
	 ${MPIRUN} -np 1 ./ex10 -f0 ${DATAFILESPATH}/matrices/medium -viewer_binary_skip_info \
                    -mat_type seqbaij -matload_block_size $$bs -ksp_max_it 100 -ksp_gmres_cgs_refinement_type refine_always -ksp_rtol \
                    1.0e-15 -ksp_smonitor >> ex10_7.tmp 2>&1 ; \
	 ${MPIRUN} -np 1 ./ex10 -f0 ${DATAFILESPATH}/matrices/medium -viewer_binary_skip_info \
                    -mat_type seqbaij -matload_block_size $$bs -ksp_max_it 100 -ksp_gmres_cgs_refinement_type refine_always -ksp_rtol \
                    1.0e-15 -ksp_smonitor -pc_factor_mat_ordering_type nd >> ex10_7.tmp 2>&1 ; \
	 ${MPIRUN} -np 1 ./ex10 -f0 ${DATAFILESPATH}/matrices/medium -viewer_binary_skip_info \
                    -mat_type seqbaij -matload_block_size $$bs -ksp_max_it 100 -ksp_gmres_cgs_refinement_type refine_always -ksp_rtol \
                    1.0e-15 -ksp_smonitor -pc_factor_levels 1 >> ex10_7.tmp 2>&1 ; \
	 ${MPIRUN} -np 1 ./ex10 -f0 ${DATAFILESPATH}/matrices/medium -viewer_binary_skip_info \
                    -mat_type seqbaij -matload_block_size $$bs -ksp_type preonly \
                    -ksp_smonitor -pc_type lu  >> ex10_7.tmp 2>&1 ; \
         done;
	-@if (${DIFF} output/ex10_7.out ex10_7.tmp) then true; \
	   else echo "Possible problem with ex10_7, diffs above"; fi;
	-@${RM} -f ex10_7.tmp
runex10_8:
	-@${MPIRUN} -np 1 ./ex10 -ksp_diagonal_scale -pc_type eisenstat -ksp_smonitor -ksp_diagonal_scale_fix \
	   -f0 ${DATAFILESPATH}/matrices/medium -ksp_gmres_cgs_refinement_type refine_always -options_left no > ex10_8.tmp 2>&1; \
	   if (${DIFF} output/ex10_8.out ex10_8.tmp) then true; \
	   else echo "Possible problem with ex10_8, diffs above"; fi; \
	   ${RM} -f ex10_8.tmp
runex10_9:
	-@touch ex10_9.tmp
	-@for type in gmres bicg bcgs tcqmr tfqmr; do \
          for bs in 1 2 3 4 5 6 7; do \
	 ${MPIRUN} -np 1 ./ex10 -f0 ${DATAFILESPATH}/matrices/medium -viewer_binary_skip_info \
                    -mat_type seqbaij -matload_block_size $$bs -ksp_max_it 100 -ksp_gmres_cgs_refinement_type refine_always -ksp_rtol \
                    1.0e-15 -ksp_smonitor >> ex10_9.tmp 2>&1 ; ${WIN_SLEEP} \
	 ${MPIRUN} -np 1 ./ex10 -f0 ${DATAFILESPATH}/matrices/medium -ksp_gmres_cgs_refinement_type refine_always -viewer_binary_skip_info \
                    -mat_type seqbaij -matload_block_size $$bs -ksp_max_it 100 -ksp_rtol \
                    1.0e-15 -ksp_smonitor -trans >> ex10_9.tmp 2>&1 ; ${WIN_SLEEP} \
          for np in 2 3; do \
	 ${MPIRUN} -np $$np ./ex10 -f0 ${DATAFILESPATH}/matrices/medium -viewer_binary_skip_info \
                    -mat_type mpibaij -matload_block_size $$bs -ksp_max_it 100 -ksp_gmres_cgs_refinement_type refine_always -ksp_rtol \
                    1.0e-15 -ksp_smonitor >> ex10_9.tmp 2>&1 ; ${WIN_SLEEP} \
	 ${MPIRUN} -np $$np ./ex10 -f0 ${DATAFILESPATH}/matrices/medium -ksp_gmres_cgs_refinement_type refine_always -viewer_binary_skip_info \
                    -mat_type mpibaij -matload_block_size $$bs -ksp_max_it 100 -ksp_rtol \
                    1.0e-15 -ksp_smonitor -trans >> ex10_9.tmp 2>&1 ; ${WIN_SLEEP} \
         done; done; done; 
	-@if (${DIFF} output/ex10_9.out ex10_9.tmp) then true; \
	   else echo "Possible problem with ex10_9, diffs above"; fi;
	-@${RM} -f ex10_9.tmp
runex10_10:
	-@${MPIRUN} -np 2 ./ex10 -options_left no -ksp_type fgmres -pc_type ksp \
	   -f0 ${DATAFILESPATH}/matrices/medium -ksp_fgmres_modifypcksp -ksp_smonitor> ex10_10.tmp 2>&1; \
	   if (${DIFF} output/ex10_10.out ex10_10.tmp) then true; \
	   else echo "Possible problem with ex10_10, diffs above"; fi; \
	   ${RM} -f ex10_10.tmp
runex10_11:
	-@${MPIRUN} -np 2 ./ex10 -options_left no -f0 ftp://info.mcs.anl.gov/pub/petsc/matrices/testmatrix.gz > ex10_11.tmp 2>&1;\
	   if (${DIFF} output/ex10_11.out ex10_11.tmp) then true; \
	   else echo "Possible problem with ex10_11, diffs above"; fi; \
	   ${RM} -f ex10_11.tmp
runex10_12:
	-@${MPIRUN} -np 1 ./ex10 -options_left no -mat_aij_matlab -pc_type lu -f0 ${DATAFILESPATH}/matrices/arco1 > ex10_12.tmp 2>&1;\
	   if (${DIFF} output/ex10_12.out ex10_12.tmp) then true; \
	   else echo "Possible problem with ex10_12, diffs above"; fi; \
	   ${RM} -f ex10_12.tmp
runex10_13:
	-@${MPIRUN} -np 1 ./ex10 -options_left no -mat_type lusol -pc_type lu -f0 ${DATAFILESPATH}/matrices/arco1 > ex10_13.tmp 2>&1;\
	   if (${DIFF} output/ex10_13.out ex10_13.tmp) then true; \
	   else echo "Possible problem with ex10_13, diffs above"; fi; \
	   ${RM} -f ex10_13.tmp
runex10_14:
	-@${MPIRUN} -np 3 ./ex10 -options_left no -pc_type spai -f0 ${DATAFILESPATH}/matrices/medium; \
	echo "ex10_14: This program tested the SPAI preconditioner."
runex10_15:
	-@${MPIRUN} -np 3 ./ex10 -options_left no -pc_type hypre -pc_hypre_type pilut -f0 ${DATAFILESPATH}/matrices/medium > ex10_15.tmp 2>&1;\
	   if (${DIFF} output/ex10_15.out ex10_15.tmp) then true; \
	   else echo "Possible problem with ex10_15, diffs above"; fi; \
	   ${RM} -f ex10_15.tmp
runex10_16:
	-@${MPIRUN} -np 3 ./ex10 -options_left no -pc_type hypre -pc_hypre_type parasails -f0 ${DATAFILESPATH}/matrices/medium > ex10_16.tmp 2>&1;\
	   if (${DIFF} output/ex10_16.out ex10_16.tmp) then true; \
	   else echo "Possible problem with ex10_16, diffs above"; fi; \
	   ${RM} -f ex10_16.tmp
runex10_17:
	-@${MPIRUN} -np 3 ./ex10 -options_left no -pc_type hypre -pc_hypre_type boomeramg -f0 ${DATAFILESPATH}/matrices/medium > ex10_17.tmp 2>&1;\
	   if (${DIFF} output/ex10_17.out ex10_17.tmp) then true; \
	   else echo "Possible problem with ex10_17, diffs above"; fi; \
	   ${RM} -f ex10_17.tmp
runex10_18:
	-@${MPIRUN} -np 3 ./ex10 -options_left no -pc_type hypre -pc_hypre_type euclid -f0 ${DATAFILESPATH}/matrices/medium > ex10_18.tmp 2>&1;\
	   if (${DIFF} output/ex10_18.out ex10_18.tmp) then true; \
	   else echo "Possible problem with ex10_18, diffs above"; fi; \
	   ${RM} -f ex10_18.tmp
LEVELS = 0 2 4
runex10_19:
	-@touch ex10_19aij.tmp
	-@touch ex10_19sbaij.tmp
	-@for levels in ${LEVELS}; do \
	${MPIRUN} -np 1 ./ex10 -f0 ${DATAFILESPATH}/matrices/poisson1 -ksp_type cg -pc_type icc -pc_factor_levels $$levels >> ex10_19aij.tmp 2>&1; \
	 ${MPIRUN} -np 1 ./ex10 -f0 ${DATAFILESPATH}/matrices/poisson1 -ksp_type cg -pc_type icc -pc_factor_levels $$levels -matload_type seqsbaij >> ex10_19sbaij.tmp 2>&1; \
	done; 
	-@if (${DIFF} ex10_19aij.tmp ex10_19sbaij.tmp) then true; \
	   else echo "Possible problem with ex10_19, diffs above"; fi; \
	${RM} -f ex10_19aij.tmp ex10_19sbaij.tmp

runex10_20: # test ilu with shift
	-@${MPIRUN} -np 1 ./ex10 -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -3.0 -pc_factor_shift_nonzero 0.1 > ex10_20.tmp 2>&1; \
	   if (${DIFF} output/ex10_20shiftnz.out ex10_20.tmp) then true; \
	   else echo "Possible problem with ex10_20shiftnz, diffs above"; fi; \
	   ${RM} -f ex10_20.tmp
	-@${MPIRUN} -np 1 ./ex10 -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -3.0 -pc_factor_shift_positive_definite > ex10_20.tmp 2>&1; \
	   if (${DIFF} output/ex10_20shiftpd.out ex10_20.tmp) then true; \
	   else echo "Possible problem with ex10_20shiftpd, diffs above"; fi; \
	   ${RM} -f ex10_20.tmp
runex10_21: #test icc with shift
	-@${MPIRUN} -np 1 ./ex10 -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -3.0 -ksp_type cg -pc_type icc -pc_factor_shift_nonzero 0.1 > ex10_21.tmp 2>&1; \
	   if (${DIFF} output/ex10_21shiftnz.out ex10_21.tmp) then true; \
	   else echo "Possible problem with ex10_21shiftnz on aij matrix, diffs above"; fi; \
	   ${RM} -f ex10_21.tmp
	-@${MPIRUN} -np 1 ./ex10 -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -3.0 -ksp_type cg -pc_type icc -pc_factor_shift_nonzero 0.1 -mat_type baij > ex10_21.tmp 2>&1; \
	   if (${DIFF} output/ex10_21shiftnz.out ex10_21.tmp) then true; \
	   else echo "Possible problem with ex10_21shiftnz on baij matrix, diffs above"; fi; \
	   ${RM} -f ex10_21.tmp
	-@${MPIRUN} -np 1 ./ex10 -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -3.0 -ksp_type cg -pc_type icc -pc_factor_shift_nonzero 0.1 -mat_type sbaij > ex10_21.tmp 2>&1; \
	   if (${DIFF} output/ex10_21shiftnz.out ex10_21.tmp) then true; \
	   else echo "Possible problem with ex10_21shiftnz on sbaij matrix, diffs above"; fi; \
	   ${RM} -f ex10_21.tmp
	-@${MPIRUN} -np 1 ./ex10 -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -3.0 -ksp_type cg -pc_type icc -pc_factor_shift_positive_definite > ex10_21.tmp 2>&1; \
	   if (${DIFF} output/ex10_21shiftpd.out ex10_21.tmp) then true; \
	   else echo "Possible problem with ex10_21shiftpd on aij matrix, diffs above"; fi; \
	   ${RM} -f ex10_21.tmp
	-@${MPIRUN} -np 1 ./ex10 -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -3.0 -ksp_type cg -pc_type icc -pc_factor_shift_positive_definite -mat_type baij > ex10_21.tmp 2>&1; \
	   if (${DIFF} output/ex10_21shiftpd.out ex10_21.tmp) then true; \
	   else echo "Possible problem with ex10_21shiftpd on baij matrix, diffs above"; fi; \
	   ${RM} -f ex10_21.tmp
	-@${MPIRUN} -np 1 ./ex10 -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -3.0 -ksp_type cg -pc_type icc -pc_factor_shift_positive_definite -mat_type sbaij > ex10_21.tmp 2>&1; \
	   if (${DIFF} output/ex10_21shiftpd.out ex10_21.tmp) then true; \
	   else echo "Possible problem with ex10_21shiftpd on sbaij matrix, diffs above"; fi; \
	   ${RM} -f ex10_21.tmp

#test mumps lu      
runex10_mumps_lu_1:
	-@${MPIRUN} -np 1 ./ex10 -f0 ${DATAFILESPATH}/matrices/small -ksp_type preonly -pc_type lu -mat_type aijmumps -num_numfac 2 -num_rhs 2 > ex10_mumps_lu_1.tmp 2>&1; \
	   if (${DIFF} output/ex10_mumps.out ex10_mumps_lu_1.tmp) then true; \
	   else echo "Possible problem with ex10_mumps_lu_1, diffs above"; fi; \
	   ${RM} -f ex10_mumps_lu_1.tmp
runex10_mumps_lu_2:
	-@${MPIRUN} -np 2 ./ex10 -f0 ${DATAFILESPATH}/matrices/small -ksp_type preonly -pc_type lu -mat_type aijmumps -num_numfac 2 -num_rhs 2 > ex10_mumps_lu_2.tmp 2>&1; \
	   if (${DIFF} output/ex10_mumps.out ex10_mumps_lu_2.tmp) then true; \
	   else echo "Possible problem with ex10_mumps_lu_2, diffs above"; fi; \
	   ${RM} -f ex10_mumps_lu_2.tmp
#test mumps cholesky 
runex10_mumps_cholesky_1:
	-@${MPIRUN} -np 1 ./ex10 -f0 ${DATAFILESPATH}/matrices/small -ksp_type preonly -pc_type cholesky -mat_type sbaijmumps -num_numfac 2 -num_rhs 2 -mat_ignore_lower_triangular > ex10_mumps_cholesky_1.tmp 2>&1; \
	   if (${DIFF} output/ex10_mumps.out ex10_mumps_cholesky_1.tmp) then true; \
	   else echo "Possible problem with ex10_mumps_cholesky_1, diffs above"; fi; \
	   ${RM} -f ex10_mumps_cholesky_1.tmp
runex10_mumps_cholesky_2:
	-@${MPIRUN} -np 2 ./ex10 -f0 ${DATAFILESPATH}/matrices/small -ksp_type preonly -pc_type cholesky -mat_type sbaijmumps -num_numfac 2 -num_rhs 2 -mat_ignore_lower_triangular > ex10_mumps_cholesky_2.tmp 2>&1; \
	   if (${DIFF} output/ex10_mumps.out ex10_mumps_cholesky_2.tmp) then true; \
	   else echo "Possible problem with ex10_mumps_cholesky_2, diffs above"; fi; \
	   ${RM} -f ex10_mumps_cholesky_2.tmp

runex11:
	-@${MPIRUN} -np 1 ./ex11 -n 6 -norandom -pc_type none -ksp_smonitor -ksp_gmres_cgs_refinement_type refine_always > ex11_1.tmp 2>&1; \
	   if (${DIFF} output/ex11_1.out ex11_1.tmp) then true; \
	   else echo "Possible problem with ex11_1, diffs above"; fi; \
	   ${RM} -f ex11_1.tmp
runex11f:
	-@${MPIRUN} -np 1 ./ex11f -n 6 -norandom -pc_type none -ksp_smonitor -ksp_gmres_cgs_refinement_type refine_always > ex11f_1.tmp 2>&1; \
	   if (${DIFF} output/ex11f_1.out ex11f_1.tmp) then true; \
	   else echo "Possible problem with ex11f_1, diffs above"; fi; \
	   ${RM} -f ex11f_1.tmp
runex12:
	-@${MPIRUN} -np 1 ./ex12 -ksp_gmres_cgs_refinement_type refine_always > ex12_1.tmp 2>&1; \
	   if (${DIFF} output/ex12_1.out ex12_1.tmp) then true; \
	   else echo "Possible problem with ex12_1, diffs above"; fi; \
	   ${RM} -f ex12_1.tmp
runex13:
	-@${MPIRUN} -np 1 ./ex13 -m 19 -n 20 -ksp_gmres_cgs_refinement_type refine_always > ex13_1.tmp 2>&1; \
	   if (${DIFF} output/ex13_1.out ex13_1.tmp) then true; \
	   else echo "Possible problem with ex13_1, diffs above"; fi; \
	   ${RM} -f ex13_1.tmp
runex13f90:
	-@${MPIRUN} -np 1 ./ex13f90 -m 19 -n 20 -ksp_gmres_cgs_refinement_type refine_always > ex13f90_1.tmp 2>&1; \
	   if (${DIFF} output/ex13f90_1.out ex13f90_1.tmp) then true; \
	   else echo "Possible problem with ex13f90_1, diffs above"; fi; \
	   ${RM} -f ex13f90_1.tmp
runex14f:
	-@${MPIRUN} -np 1 ./ex14f -no_output -ksp_gmres_cgs_refinement_type refine_always > ex14_1.tmp 2>&1; \
	   if (${DIFF} output/ex14_1.out ex14_1.tmp) then true; \
	   else echo "Possible problem with ex14f_1, diffs above"; fi; \
	   ${RM} -f ex14_1.tmp
runex15:
	-@${MPIRUN} -np 2 ./ex15 -ksp_view -user_defined_pc -ksp_gmres_cgs_refinement_type refine_always > ex15_1.tmp 2>&1; \
	   if (${DIFF} output/ex15_1.out ex15_1.tmp) then true; \
	   else echo "Possible problem with ex15_1, diffs above"; fi; \
	   ${RM} -f ex15_1.tmp
runex15f:
	-@${MPIRUN} -np 2 ./ex15f -ksp_view -user_defined_pc -ksp_gmres_cgs_refinement_type refine_always > ex15f_1.tmp 2>&1; \
	   if (${DIFF} output/ex15f_1.out ex15f_1.tmp) then true; \
	   else echo "Possible problem with ex15f_1, diffs above"; fi; \
	   ${RM} -f ex15f_1.tmp
runex16:
	-@${MPIRUN} -np 2 ./ex16 -ntimes 4 -ksp_gmres_cgs_refinement_type refine_always > ex16_1.tmp 2>&1; \
	   if (${DIFF} output/ex16_1.out ex16_1.tmp) then true; \
	   else echo "Possible problem with ex16_1, diffs above"; fi; \
	   ${RM} -f ex16_1.tmp
runex21f:
	-@${MPIRUN} -np 1 ./ex21f  > ex21f_1.tmp 2>&1; \
	   if (${DIFF} output/ex21f_1.out ex21f_1.tmp) then true; \
	   else echo "Possible problem with ex21f_1, diffs above"; fi; \
	   ${RM} -f ex21f_1.tmp
runex22:
	-@${MPIRUN} -np 1 ./ex22 -pc_mg_type full -dmmg_ksp_monitor > ex22_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex22_1.out ex22_1.tmp) then true; \
	   else echo "Possible problem with ex22_1, diffs above"; fi; \
	   ${RM} -f ex22_1.tmp

runex22f:
	-@${MPIRUN} -np 1 ./ex22f -pc_mg_type full -dmmg_ksp_monitor > ex22_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex22_1.out ex22_1.tmp) then true; \
	   else echo "Possible problem with ex22f_1, diffs above"; fi; \
	   ${RM} -f ex22_1.tmp

runex23:
	-@${MPIRUN} -np 1 ./ex23 -ksp_smonitor -ksp_gmres_cgs_refinement_type refine_always > ex23_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex23_1.out ex23_1.tmp) then true; \
	   else echo "Possible problem with ex23_1, diffs above"; fi; \
	   ${RM} -f ex23_1.tmp

runex23_2:
	-@${MPIRUN} -np 3 ./ex23 -ksp_smonitor -ksp_gmres_cgs_refinement_type refine_always > ex23_2.tmp 2>&1;	  \
	   if (${DIFF} output/ex23_2.out ex23_2.tmp) then true; \
	   else echo "Possible problem with ex23_2, diffs above"; fi; \
	   ${RM} -f ex23_2.tmp

runex25:
	-@${MPIRUN} -np 1 ./ex25 > ex25_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex25_1.out ex25_1.tmp) then true; \
	   else echo "Possible problem with ex25_1, diffs above"; fi; \
	   ${RM} -f ex25_1.tmp

runex25_2:
	-@${MPIRUN} -np 2 ./ex25 > ex25_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex25_1.out ex25_1.tmp) then true; \
	   else echo "Possible problem with ex25_2, diffs above"; fi; \
	   ${RM} -f ex25_1.tmp

runex29:
	-@${MPIRUN} -np 1 ./ex29 -ksp_smonitor  -dmmg_nlevels 9 > ex29_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex29_1.out ex29_1.tmp) then true; \
	   else echo "Possible problem with ex29_1, diffs above"; fi; \
	   ${RM} -f ex29_1.tmp

runex29_2:
	-@${MPIRUN} -np 1 ./ex29 -ksp_smonitor  -bc_type neumann -dmmg_nlevels 9 > ex29_2.tmp 2>&1;	  \
	   if (${DIFF} output/ex29_2.out ex29_2.tmp) then true; \
	   else echo "Possible problem with ex29_2, diffs above"; fi; \
	   ${RM} -f ex29_2.tmp

runex30:
	-@${MPIRUN} -np 1 ./ex30 > ex30_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex30_1.out ex30_1.tmp) then true; \
	   else echo "Possible problem with ex30_1, diffs above"; fi; \
	   ${RM} -f ex30_1.tmp

runex32:
	-@${MPIRUN} -np 1 ./ex32 -ksp_smonitor  > ex32_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex32_1.out ex32_1.tmp) then true; \
	   else echo "Possible problem with ex32_1, diffs above"; fi; \
	   ${RM} -f ex32_1.tmp

runex33:
	-${MPIRUN} -np ${NP} ./ex33 -dmmg_nlevels 1 -ksp_max_it 100 -mat_view -ksp_monitor -use_zero_base 1 -base_file ${PETSC_DIR}/src/dm/mesh/examples/tutorials/data/ex1_2d -bc_file data/ex33bc.data ${EXTRA_ARGS}

runex35:
	-${MPIRUN} -np ${NP} ./ex35 -dmmg_nlevels 1 -ksp_max_it 100 -mat_view -ksp_monitor -use_zero_base 1 -base_file ${PETSC_DIR}/src/dm/mesh/examples/tutorials/data/ex1_2d ${EXTRA_ARGS}


debugex33:
	-${MPIRUN} -np ${NP} ./ex33 -dmmg_nlevels 1 -ksp_max_it 100 -mat_view -ksp_monitor -use_zero_base 1 -base_file ${PETSC_DIR}/src/dm/mesh/examples/tutorials/data/ex1_2d -bc_file data/ex33bc.data ${EXTRA_ARGS} -start_in_debugger -display ${DISPLAY}

TESTEXAMPLES_C		       = ex1.PETSc runex1 runex1_2 runex1_3 ex1.rm ex2.PETSc runex2 runex2_2 runex2_3 \
                                 runex2_4 ex2.rm ex7.PETSc runex7 ex7.rm ex5.PETSc runex5 runex5_2 ex5.rm \
                                 ex9.PETSc runex9 ex9.rm ex12.PETSc runex12 ex12.rm ex13.PETSc runex13 ex13.rm \
                                 ex15.PETSc runex15 ex15.rm ex16.PETSc runex16 ex16.rm ex22.PETSc runex22 ex22.rm \
                                 ex23.PETSc runex23 runex23_2 ex23.rm  ex25.PETSc runex25 runex25_2 ex25.rm \
                                 ex27.PETSc ex27.rm ex28.PETSc ex28.rm ex29.PETSc  ex29.rm \
                                 ex31.PETSc ex31.rm ex32.PETSc runex32 ex32.rm
TESTEXAMPLES_C_X11	       = ex2.PETSc runex2_5 ex2.rm ex4.PETSc ex4.rm ex8.PETSc ex5.PETSc runex5_5 ex5.rm ex8.rm 
TESTEXAMPLES_FORTRAN	       = ex1f.PETSc runex1f ex1f.rm ex2f.PETSc runex2f ex2f.rm ex6f.PETSc ex6f.rm \
                                 ex14f.PETSc runex14f ex14f.rm ex15f.PETSc runex15f ex15f.rm ex22f.PETSc runex22f \
                                 ex22f.rm ex21f.PETSc runex21f ex21f.rm
TESTEXAMPLES_FORTRAN_MPIUNI    = ex1f.PETSc runex1f ex1f.rm ex6f.PETSc runex6f ex6f.rm
TESTEXAMPLES_C_X11_MPIUNI      = ex1.PETSc runex1 runex1_2 runex1_3 ex1.rm ex2.PETSc runex2 runex2_3 ex2.rm \
                                 ex7.PETSc ex7.rm ex5.PETSc ex5.rm ex4.PETSc ex4.rm ex9.PETSc runex9 ex9.rm \
                                 ex23.PETSc runex23 ex23.rm
TESTEXAMPLES_C_COMPLEX	       = ex10.PETSc runex10 ex10.rm ex11.PETSc runex11 ex11.rm
TESTEXAMPLES_C_NOCOMPLEX       = ex10.PETSc runex10_2 runex10_3 runex10_4 runex10_5 runex10_6 runex10_7 runex10_8 \
                                 runex10_9 runex10_10 runex10_19 runex10_20 ex10.rm
TESTEXAMPLES_BLOCKSOLVE95      = ex5.PETSc runex5_3 runex5_4 ex5.rm
# even though ex10.c is a C example to run with -mat_type lusol requires a Fortran compiler, hence
# we list it with the fortran examples
TESTEXAMPLES_FORTRAN_NOCOMPLEX = 
TESTEXAMPLES_FORTRAN_COMPLEX   = ex11f.PETSc runex11f ex11f.rm
TESTEXAMPLES_F90	       = ex13f90.PETSc runex13f90 ex13f90.rm
TESTEXAMPLES_13		       = ex3.PETSc ex3.rm ex14f.PETSc ex14f.rm
TESTEXAMPLES_MATLAB_ENGINE     = ex10.PETSc runex10_12  ex10.rm
TESTEXAMPLES_17		       = ex10.PETSc runex10_11 ex10.rm
TESTEXAMPLES_18		       = ex2.PETSc runex2_6 ex2.rm 
TESTEXAMPLES_SPAI	       = ex10.PETSc runex10_14 ex10.rm
TESTEXAMPLES_HYPRE	       = ex10.PETSc runex10_15 runex10_16 runex10_17 runex10_18 ex10.rm
TESTEXAMPLES_LUSOL	       = ex10.PETSc runex10_13 ex10.rm
TESTEXAMPLES_MUMPS             = ex10.PETSc runex10_mumps_lu_1 runex10_mumps_lu_2 runex10_mumps_cholesky_1 \
                                 runex10_mumps_cholesky_2 ex10.rm

include ${PETSC_DIR}/bmake/common/test
