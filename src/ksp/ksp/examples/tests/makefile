
CFLAGS	  = 
FFLAGS    =
CPPFLAGS  = 
FPPFLAGS  =
LOCDIR    = src/ksp/ksp/examples/tests/
NOADIC    = true
EXAMPLESC       = ex1.c ex3.c ex4.c ex6.c ex7.c ex10.c \
		  ex14.c ex15.c ex17.c ex18.c ex19.c ex20.c ex21.c ex24.c ex25.c ex26.c ex27.c ex32.c ex34.c
EXAMPLESCH      = 
EXAMPLESF       = ex5f.F ex12f.F ex16f.F

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_KSP_LIB}
	${RM} ex1.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_KSP_LIB}
	${RM} ex2.o

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

ex12f: ex12f.o  chkopts
	-${FLINKER} -o ex12f ex12f.o  ${PETSC_KSP_LIB}
	${RM} ex12f.o

ex5f: ex5f.o  chkopts
	-${FLINKER} -o ex5f ex5f.o  ${PETSC_KSP_LIB}
	${RM} ex5f.o

ex13: ex13.o  chkopts
	-${CLINKER} -o ex13 ex13.o  ${PETSC_KSP_LIB}
	${RM} ex13.o

ex14: ex14.o  chkopts
	-${CLINKER} -o ex14 ex14.o  ${PETSC_KSP_LIB}
	${RM} ex14.o

ex15: ex15.o  chkopts
	-${CLINKER} -o ex15 ex15.o  ${PETSC_KSP_LIB}
	${RM} ex15.o

ex16f: ex16f.o  chkopts
	-${FLINKER} -o ex16f ex16f.o  ${PETSC_KSP_LIB}
	${RM} ex16f.o

ex17: ex17.o  chkopts
	-${CLINKER} -o ex17 ex17.o  ${PETSC_KSP_LIB}
	${RM} ex17.o

ex18: ex18.o  chkopts
	-${CLINKER} -o ex18 ex18.o  ${PETSC_KSP_LIB}
	${RM} ex18.o

ex19: ex19.o  chkopts
	-${CLINKER} -o ex19 ex19.o  ${PETSC_KSP_LIB}
	${RM} ex19.o

ex20: ex20.o  chkopts
	-${CLINKER} -o ex20 ex20.o  ${PETSC_KSP_LIB}
	${RM} ex20.o

ex21: ex21.o  chkopts
	-${CLINKER} -o ex21 ex21.o   ${PETSC_KSP_LIB}
	${RM} ex21.o

ex22: ex22.o  chkopts
	-${CLINKER} -o ex22 ex22.o  ${PETSC_KSP_LIB}
	${RM} ex22.o

ex24: ex24.o  chkopts
	-${CLINKER} -o ex24 ex24.o  ${PETSC_KSP_LIB}
	${RM} ex24.o
ex25: ex25.o  chkopts
	-${CLINKER} -o ex25 ex25.o  ${PETSC_KSP_LIB}
	${RM} ex25.o
ex26: ex26.o  chkopts
	-${CLINKER} -o ex26 ex26.o  ${PETSC_KSP_LIB}
	${RM} ex26.o
ex27: ex27.o  chkopts
	-${CLINKER} -o ex27 ex27.o  ${PETSC_KSP_LIB}
	${RM} ex27.o
ex28: ex28.o  chkopts
	-${CLINKER} -o ex28 ex28.o  ${PETSC_KSP_LIB}
	${RM} ex28.o
ex29: ex29.o  chkopts
	-${CLINKER} -o ex29 ex29.o  ${PETSC_KSP_LIB}
	${RM} ex29.o
ex30: ex30.o  chkopts
	-${CLINKER} -o ex30 ex30.o  ${PETSC_KSP_LIB}
	${RM} ex30.o 
ex31: ex31.o  chkopts
	-${CLINKER} -o ex31 ex31.o  ${PETSC_KSP_LIB}
	${RM} ex31.o
ex32: ex32.o  chkopts
	-${CLINKER} -o ex32 ex32.o  ${PETSC_KSP_LIB}
	${RM} ex32.o
ex33: ex33.o  chkopts
	-${CLINKER} -o ex33 ex33.o  ${PETSC_KSP_LIB}
	${RM} ex33.o
ex34: ex34.o  chkopts
	-${CLINKER} -o ex34 ex34.o  ${PETSC_KSP_LIB}
	${RM} ex34.o
#------------------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 -pc_type jacobi -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex1_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   ${RM} -f ex1_1.tmp
runex1_2:
	-@${MPIEXEC} -n 2 ./ex1 -pc_type jacobi -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex1_2.tmp 2>&1;	 \
	   if (${DIFF} output/ex1_2.out ex1_2.tmp) then true; \
	   else echo "Possible problem with ex1_2, diffs above"; fi; \
	   ${RM} -f ex1_2.tmp
runex1_3:
	-@${MPIEXEC} -n 1 ./ex1 -pc_type sor -pc_sor_symmetric -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always >\
	   ex1_3.tmp 2>&1;   \
	   if (${DIFF} output/ex1_3.out ex1_3.tmp) then true; \
	   else echo "Possible problem with ex1_3, diffs above"; fi; \
	   ${RM} -f ex1_3.tmp
runex1_5:
	-@${MPIEXEC} -n 1 ./ex1 -pc_type eisenstat -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always >\
	   ex1_5.tmp 2>&1;   \
	   if (${DIFF} output/ex1_5.out ex1_5.tmp) then true; \
	   else echo "Possible problem with ex1_5, diffs above"; fi; \
	   ${RM} -f ex1_5.tmp
runex3:
	-@${MPIEXEC} -n 1 ./ex3 -pc_type jacobi -ksp_monitor_short -m 5 -ksp_gmres_cgs_refinement_type refine_always > ex3_1.tmp 2>&1;   \
	   if (${DIFF} output/ex3_1.out ex3_1.tmp) then true; \
	   else echo "Possible problem with ex3_1, diffs above"; fi; \
	   ${RM} -f ex3_1.tmp
runex3_2:
	-@${MPIEXEC} -n 2 ./ex3 -pc_type jacobi -ksp_monitor_short -m 5 -ksp_gmres_cgs_refinement_type refine_always > ex3_2.tmp 2>&1;   \
	   if (${DIFF} output/ex3_2.out ex3_2.tmp) then true; \
	   else echo "Possible problem with ex3_2, diffs above"; fi; \
	   ${RM} -f ex3_2.tmp
runex4:
	-@${MPIEXEC} -n 1 ./ex4 -ksp_monitor_short -m 5 -pc_type jacobi -ksp_gmres_cgs_refinement_type refine_always > ex4_1.tmp 2>&1;\
	   if (${DIFF} output/ex4_1.out ex4_1.tmp) then true; \
	   else echo "Possible problem with ex4_1, diffs above"; fi; \
	   ${RM} -f ex4_1.tmp
runex4_3:
	-@${MPIEXEC} -n 1 ./ex4 -pc_type sor -pc_sor_symmetric -ksp_monitor_short -m 5 -ksp_gmres_cgs_refinement_type refine_always >\
	   ex4_3.tmp 2>&1;   \
	   if (${DIFF} output/ex4_3.out ex4_3.tmp) then true; \
	   else echo "Possible problem with ex4_3, diffs above"; fi; \
	   ${RM} -f ex4_3.tmp
runex4_5:
	-@${MPIEXEC} -n 1 ./ex4 -pc_type eisenstat -ksp_monitor_short -m 5	-ksp_gmres_cgs_refinement_type refine_always >\
	   ex4_5.tmp 2>&1;   \
	   if (${DIFF} output/ex4_5.out ex4_5.tmp) then true; \
	   else echo "Possible problem with ex4_5, diffs above"; fi; \
	   ${RM} -f ex4_5.tmp
runex5:
	-@${MPIEXEC} -n 2 ./ex5 -ksp_monitor_short -nokspview -ksp_gmres_cgs_refinement_type refine_always > ex5_1.tmp 2>&1; \
	   if (${DIFF} output/ex5_1.out ex5_1.tmp) then true; \
	   else echo "Possible problem with ex5_1, diffs above"; fi; \
	   ${RM} -f ex5_1.tmp
runex5f:
	-@${MPIEXEC} -n 1 ./ex5f -ksp_monitor_short > ex5f_1.tmp 2>&1; \
	   if (${DIFF} output/ex5f_1.out ex5f_1.tmp) then true; \
	   else echo "Possible problem with ex5f_1, diffs above"; fi; \
	   ${RM} -f ex5f_1.tmp
#
# See http://www.mcs.anl.gov/petsc/petsc-as/documentation/faq.html#datafiles for how to obtain the datafiles used below
runex6:
	-@${MPIEXEC} -n 1 ./ex6 -ksp_type preonly  -pc_type lu -options_left no \
	   -f ${DATAFILESPATH}/matrices/arco1 > ex6_1.tmp 2>&1; \
	   if (${DIFF} output/ex6_1.out ex6_1.tmp) then true; \
	   else echo "Possible problem with ex6_1, diffs above"; fi; \
	   ${RM} -f ex6_1.tmp
runex6_2:
	-@${MPIEXEC} -n 1 ./ex6 -sub_pc_type ilu -options_left no \
	   -f ${DATAFILESPATH}/matrices/arco1 \
	   -ksp_gmres_restart 100 -ksp_gmres_cgs_refinement_type refine_always -sub_ksp_type preonly -pc_type bjacobi \
	   -pc_bjacobi_blocks 8 -sub_pc_factor_in_place -ksp_monitor_short \
	   > ex6_2.tmp 2>&1; \
	   if (${DIFF} output/ex6_2.out ex6_2.tmp) then true; \
	   else echo "Possible problem with ex6_2, diffs above"; fi; \
	   ${RM} -f ex6_2.tmp
runex6_3:
	-@${MPIEXEC} -n 1 ./ex6 -pc_type ilu -options_left no \
	   -f ${DATAFILESPATH}/matrices/arco4 \
	   -ksp_gmres_restart 100  -ksp_gmres_cgs_refinement_type refine_always -ksp_monitor_short  > ex6_3.tmp 2>&1; \
	   if (${DIFF} output/ex6_3.out ex6_3.tmp) then true; \
	   else echo "Possible problem with ex6_3, diffs above"; fi; \
	   ${RM} -f ex6_3.tmp

runex6_7:
	-@${MPIEXEC} -n 1 ./ex6 -ksp_gmres_cgs_refinement_type refine_always -pc_type asm -pc_asm_blocks 6 \
	   -f ${DATAFILESPATH}/matrices/small \
	   -matload_block_size 6  -ksp_monitor_short  > ex6_7.tmp 2>&1; \
	   if (${DIFF} output/ex6_7.out ex6_7.tmp) then true; \
	   else echo "Possible problem with ex6_7, diffs above"; fi; \
	   ${RM} -f ex6_7.tmp
runex7:
	-@${MPIEXEC} -n 2 ./ex7 -ksp_gmres_cgs_refinement_type refine_always -pc_type jacobi -ksp_monitor_short > ex7_1.tmp 2>&1; \
	   if (${DIFF} output/ex7_1.out ex7_1.tmp) then true; \
	   else echo "Possible problem with ex7_1, diffs above"; fi; \
	   ${RM} -f ex7_1.tmp
runex8:
	-@${MPIEXEC} -n 1 ./ex8 -ksp_gmres_cgs_refinement_type refine_always -pc_type jacobi -ksp_monitor_short > ex8_1.tmp 2>&1; \
	   if (${DIFF} output/ex8_1.out ex8_1.tmp) then true; \
	   else echo "Possible problem with ex8_1, diffs above"; fi; \
	   ${RM} -f ex8_1.tmp
runex8_2:
	-@${MPIEXEC} -n 2 ./ex8 -ksp_gmres_cgs_refinement_type refine_always -pc_type jacobi -ksp_monitor_short \
	   -ksp_rtol .000001 > ex8_2.tmp 2>&1;   \
	   if (${DIFF} output/ex8_2.out ex8_2.tmp) then true; \
	   else echo "Possible problem with ex8_2, diffs above"; fi; \
	   ${RM} -f ex8_2.tmp

runex10:
	-@${MPIEXEC} -n 1 ./ex10 -matconvert_type seqaij -ksp_monitor_short >ex10_1.tmp 2>&1; \
	   if (${DIFF} output/ex10_1.out ex10_1.tmp) then true; \
	   else echo "Possible problem with ex10_1, diffs above"; fi; \
	   ${RM} -f ex10_1.tmp
runex11:
	-@${MPIEXEC} -n 3  ./ex11 -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex11_1.tmp 2>&1; \
	   if (${DIFF} output/ex11_1.out ex11_1.tmp) then true; \
	   else echo "Possible problem with ex11_1, diffs above"; fi; \
	   ${RM} -f ex11_1.tmp
runex12f:
	-@${MPIEXEC} -n 1 ./ex12f  -f ../../../../mat/examples/matbinary.ex -options_left no > ex12f_1.tmp 2>&1; \
	   if (${DIFF} output/ex12f_1.out ex12f_1.tmp) then true; \
	   else echo "Possible problem with ex12f_1, diffs above"; fi; \
	   ${RM} -f ex12f_1.tmp
runex13:
	-@${MPIEXEC} -n 2  ./ex13 -ksp_gmres_cgs_refinement_type refine_always -ksp_monitor_short > ex13_1.tmp 2>&1; \
	   if (${DIFF} output/ex13_1.out ex13_1.tmp) then true; \
	   else echo "Possible problem with ex13_1, diffs above"; fi; \
	   ${RM} -f ex13_1.tmp
runex16f:
	-@${MPIEXEC} -n 1 ./ex16f  -f ../../../../mat/examples/matbinary.ex -options_left no> ex16f_1.tmp 2>&1; \
	   if (${DIFF} output/ex16f_1.out ex16f_1.tmp) then true; \
	   else echo "Possible problem with ex16f_1, diffs above"; fi; \
	   ${RM} -f ex16f_1.tmp
runex17:
	-@${MPIEXEC} -n 1 ./ex17 -ksp_gmres_cgs_refinement_type refine_always -n 6 -ksp_monitor_short -p 5 -norandom -ksp_type gmres \
	   -pc_type jacobi > ex17_1.tmp 2>&1; \
	   if (${DIFF} output/ex17_1.out ex17_1.tmp) then true; \
	   else echo "Possible problem with ex17_1, diffs above"; fi; \
	   ${RM} -f ex17_1.tmp
runex18:
	-@${MPIEXEC} -n 1 ./ex18 -ksp_gmres_cgs_refinement_type refine_always -f  ../../../../mat/examples/matbinary.ex
runex19:
	-@${MPIEXEC} -n 1 ./ex19 -ksp_gmres_cgs_refinement_type refine_always -pc_type jacobi -ksp_monitor_short \
	-ksp_type gmres  > ex19_1.tmp 2>&1; \
	   if (${DIFF} output/ex19_1.out ex19_1.tmp) then true; \
	   else echo "Possible problem with ex19_1, diffs above"; fi; \
	   ${RM} -f ex19_1.tmp
runex19_2:
	-@${MPIEXEC} -n 3 ./ex19 -ksp_monitor_short > ex19_2.tmp 2>&1; \
	if (${DIFF} output/ex19_2.out ex19_2.tmp) then true; \
	   else echo "Possible problem with ex19_2, diffs above"; fi; \
	   ${RM} -f ex19_2.tmp
runex20:
	-@${MPIEXEC} -n 1 ./ex20 -ksp_gmres_cgs_refinement_type refine_always -pc_type jacobi -mat_view -ksp_monitor_short -ksp_atol 1.0e-13 > ex20_1.tmp 2>&1; \
	   if (${DIFF} output/ex20_1.out ex20_1.tmp) then true; \
	   else echo "Possible problem with ex20_1, diffs above"; fi; \
	   ${RM} -f ex20_1.tmp

runex21:
	-@${MPIEXEC} -n 1 ./ex21 > ex21_1.tmp 2>&1; \
	   if (${DIFF} output/ex21_1.out ex21_1.tmp) then true; \
	   else echo "Possible problem with ex21_1, diffs above"; fi; \
	   ${RM} -f ex21_1.tmp

runex21_2:
	-@${MPIEXEC} -n 2 ./ex21 > ex21_2.tmp 2>&1; \
	   if (${DIFF} output/ex21_2.out ex21_2.tmp) then true; \
	   else echo "Possible problem with ex21_2, diffs above"; fi; \
	   ${RM} -f ex21_2.tmp


runex21_3:
	-@${MPIEXEC} -n 3 ./ex21 > ex21_3.tmp 2>&1; \
	   if (${DIFF} output/ex21_3.out ex21_3.tmp) then true; \
	   else echo "Possible problem with ex21_3, diffs above"; fi; \
	   ${RM} -f ex21_3.tmp

runex22:
	-@${MPIEXEC} -n 1 ./ex22 -ksp_gmres_cgs_refinement_type refine_always -mat_type seqaij -mat_view_info_detailed -ksp_monitor_short > ex22_1.tmp 2>&1; \
	   if (${DIFF} output/ex22_1.out ex22_1.tmp) then true; \
	   else echo "Possible problem with ex22_1, diffs above"; fi; \
	   ${RM} -f ex22_1.tmp

runex22_2:
	-@${MPIEXEC} -n 4 ./ex22 -ksp_gmres_cgs_refinement_type refine_always -m_eta 10 -m_xi 11 -mat_type seqaij -mat_view_info \
	   -ksp_monitor_short -pc_type bjacobi > ex22_2.tmp 2>&1;	  \
	   if (${DIFF} output/ex22_2.out ex22_2.tmp) then true; \
	   else echo "Possible problem with ex22_2, diffs above"; fi; \
	   ${RM} -f ex22_2.tmp
runex24:
	-@${MPIEXEC} -n 1 ./ex24 -pc_type icc -mat_type seqsbaij -mat_ignore_lower_triangular > ex24_1.tmp 2>&1; \
           if (${DIFF} output/ex24_1.out ex24_1.tmp) then true; \
           else echo "Possible problem with ex24_1, diffs above"; fi; \
           ${RM} -f ex24_1.tmp
runex24_2:
	-@${MPIEXEC} -n 1 ./ex24 -pc_type icc -pc_factor_levels 2  -mat_type seqsbaij -mat_ignore_lower_triangular > ex24_2.tmp 2>&1; \
           if (${DIFF} output/ex24_2.out ex24_2.tmp) then true; \
           else echo "Possible problem with ex24_2, diffs above"; fi; \
           ${RM} -f ex24_2.tmp
runex24_3:
	-@${MPIEXEC} -n 2 ./ex24 -pc_type bjacobi -sub_pc_type icc  -mat_type mpisbaij -mat_ignore_lower_triangular > ex24_3.tmp 2>&1; \
           if (${DIFF} output/ex24_3.out ex24_3.tmp) then true; \
           else echo "Possible problem with ex24_3, diffs above"; fi; \
           ${RM} -f ex24_3.tmp
runex24_4:
	-@${MPIEXEC} -n 2 ./ex24 -pc_type bjacobi -sub_pc_type icc -sub_pc_factor_levels 1 -mat_type mpisbaij -mat_ignore_lower_triangular > ex24_4.tmp 2>&1; \
           if (${DIFF} output/ex24_4.out ex24_4.tmp) then true; \
           else echo "Possible problem with ex24_4, diffs above"; fi; \
           ${RM} -f ex24_4.tmp
# See http://www.mcs.anl.gov/petsc/petsc-as/documentation/faq.html#datafiles for how to obtain the datafiles used below
runex25:
	-@${MPIEXEC} -n 1 ./ex25 -fload ${DATAFILESPATH}/matrices/indefinite/afiro > ex25_1.tmp 2>&1; \
           if (${DIFF} output/ex25_1.out ex25_1.tmp) then true; \
           else echo "Possible problem with ex25_1, diffs above"; fi; \
           ${RM} -f ex25_1.tmp
runex25_2:
	-@${MPIEXEC} -n 1 ./ex25 -fload ${DATAFILESPATH}/matrices/indefinite/afiro \
		-pc_type jacobi -pc_jacobi_rowmax > ex25_2.tmp 2>&1; \
           if (${DIFF} output/ex25_2.out ex25_2.tmp) then true; \
           else echo "Possible problem with ex25_2, diffs above"; fi; \
           ${RM} -f ex25_2.tmp 
runex26:
	-@${MPIEXEC} -n 1 ./ex26 -ksp_monitor_short > ex26_1.tmp 2>&1; \
	if (${DIFF} output/ex26_1.out ex26_1.tmp) then true; \
	   else echo "Possible problem with ex26_1, diffs above"; fi; \
	   ${RM} -f ex26_1.tmp
runex26_2:
	-@${MPIEXEC} -n 3 ./ex26 -ksp_monitor_short > ex26_2.tmp 2>&1; \
	if (${DIFF} output/ex26_2.out ex26_2.tmp) then true; \
	   else echo "Possible problem with ex26_2, diffs above"; fi; \
	   ${RM} -f ex26_2.tmp

runex26_ml:
	-@${MPIEXEC} -n 1 ./ex26 -ksp_monitor_short -pc_type ml -mat_no_inode > ex26_1.tmp 2>&1; \
	if (${DIFF} output/ex26_ml_1.out ex26_1.tmp) then true; \
           else echo "Possible problem with ex26_ml_1, diffs above"; fi; \
           ${RM} -f ex26_1.tmp
runex26_ml_2:
	-@${MPIEXEC} -n 3 ./ex26 -ksp_monitor_short -pc_type ml -mat_no_inode > ex26_2.tmp 2>&1; \
	if (${DIFF} output/ex26_ml_2.out ex26_2.tmp) then true; \
	   else echo "Possible problem with ex26_ml_2, diffs above"; fi; \
	   ${RM} -f ex26_2.tmp
runex26_ml_3:
	-@${MPIEXEC} -n 1 ./ex26 -ksp_monitor_short -pc_type ml -mat_no_inode -pc_mg_type ADDITIVE > ex26_2.tmp 2>&1; \
	if (${DIFF} output/ex26_ml_3.out ex26_2.tmp) then true; \
	   else echo "Possible problem with ex26_ml_3, diffs above"; fi; \
	   ${RM} -f ex26_2.tmp
# See http://www.mcs.anl.gov/petsc/petsc-as/documentation/faq.html#datafiles for how to obtain the datafiles used below
runex27:
	-@${MPIEXEC} -n 1 ./ex27 -f ${DATAFILESPATH}/matrices/small > ex27.tmp 2>&1; \
	if (${DIFF} output/ex27.out ex27.tmp) then true; \
	   else echo "Possible problem with ex27, diffs above"; fi; \
	   ${RM} -f ex27.tmp
runex29:
	-@${MPIEXEC} -n 1 ./ex29 -ksp_monitor_short -mat_no_inode > ex29.tmp 2>&1; \
	if (${DIFF} output/ex29.out ex29.tmp) then true; \
	   else echo "Possible problem with ex29, diffs above"; fi; \
	   ${RM} -f ex29.tmp
runex29_2:
	-@${MPIEXEC} -n 3 ./ex29 -ksp_monitor_short > ex29_2.tmp 2>&1; \
	if (${DIFF} output/ex29_2.out ex29_2.tmp) then true; \
	   else echo "Possible problem with ex29_2, diffs above"; fi; \
	   ${RM} -f ex29_2.tmp
runex30:
	-@${MPIEXEC} -n 1 ./ex30 -f ${DATAFILESPATH}/matrices/small -ksp_type preonly -pc_type ilu -pc_factor_drop_tolerance 0.0,0.0,4 -pc_factor_mat_ordering_type natural -num_numfac 2 -pc_factor_reuse_fill > ex30.tmp 2>&1; \
	if (${DIFF} output/ex30.out ex30.tmp) then true; \
	   else echo "Possible problem with ex30, diffs above"; fi; \
	   ${RM} -f ex30.tmp
runex30_2:
	-@${MPIEXEC} -n 1 ./ex30 -f ${DATAFILESPATH}/matrices/arco1 -mat_type baij -matload_block_size 3 -ksp_type preonly -pc_type ilu -pc_factor_drop_tolerance 0.0,0.0,50 -pc_factor_mat_ordering_type natural -num_numfac 2 > ex30.tmp 2>&1; \
	if (${DIFF} output/ex30_2.out ex30.tmp) then true; \
	   else echo "Possible problem with ex30_2, diffs above"; fi; \
	   ${RM} -f ex30.tmp
runex30_3:
	-@${MPIEXEC} -n 1 ./ex30 -f ${DATAFILESPATH}/matrices/arco1 -mat_type baij -matload_block_size 3 -ksp_type preonly -pc_type ilu -pc_factor_drop_tolerance 1.e-4,0.0,50 -pc_factor_mat_ordering_type natural -num_numfac 2 > ex30.tmp 2>&1; \
        if (${DIFF} output/ex30_3.out ex30.tmp) then true; \
           else echo "Possible problem with ex30_3, diffs above"; fi; \
           ${RM} -f ex30.tmp

runex30_shift: # test lu with shift
	-@${MPIEXEC} -n 1 ./ex30 -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type lu -pc_factor_shift_type NONZERO -pc_factor_shift_amount 1.e-5 > ex30.tmp 2>&1; \
	   if (${DIFF} output/ex30_shiftnz.out ex30.tmp) then true; \
	   else echo "Possible problem with ex30_shiftnz, diffs above"; fi; \
	   ${RM} -f ex30.tmp
	-@${MPIEXEC} -n 1 ./ex30 -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type lu -pc_factor_shift_type POSITIVE_DEFINITE > ex30.tmp 2>&1; \
	   if (${DIFF} output/ex30_shiftpd.out ex30.tmp) then true; \
	   else echo "Possible problem with ex30_shiftpd, diffs above"; fi; \
	   ${RM} -f ex30.tmp

runex30_shift_cholesky_aij: # test cholesky with shift
	-@${MPIEXEC} -n 1 ./ex30 -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type cholesky -pc_factor_shift_type NONZERO -pc_factor_shift_amount 1.e-5 > ex30.tmp 2>&1; \
	   if (${DIFF} output/ex30_shiftnz.out ex30.tmp) then true; \
	   else echo "Possible problem with ex30_shiftnz_cholesky_aij, diffs above"; fi; \
	   ${RM} -f ex30.tmp
	-@${MPIEXEC} -n 1 ./ex30 -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type cholesky -pc_factor_shift_type POSITIVE_DEFINITE > ex30.tmp 2>&1; \
	   if (${DIFF} output/ex30_shiftpd_2.out ex30.tmp) then true; \
	   else echo "Possible problem with ex30_shiftpd_cholesky_aij, diffs above"; fi; \
	   ${RM} -f ex30.tmp

runex30_shift_cholesky_sbaij: # test cholesky with shift
	-@${MPIEXEC} -n 1 ./ex30 -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type cholesky -pc_factor_shift_type NONZERO -pc_factor_shift_amount 1.e-5 -mat_type sbaij > ex30.tmp 2>&1; \
	   if (${DIFF} output/ex30_shiftnz.out ex30.tmp) then true; \
	   else echo "Possible problem with ex30_shiftnz_cholesky_sbaij, diffs above"; fi; \
	   ${RM} -f ex30.tmp
	-@${MPIEXEC} -n 1 ./ex30 -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type cholesky -pc_factor_shift_type POSITIVE_DEFINITE -mat_type sbaij > ex30.tmp 2>&1; \
	   if (${DIFF} output/ex30_shiftpd_2.out ex30.tmp) then true; \
	   else echo "Possible problem with ex30_shiftpd_cholesky_sbaij, diffs above"; fi; \
	   ${RM} -f ex30.tmp

runex30_shift_in_blocks_aij: # sensitive to machine, thus not submitted to the nightly tests
	-@${MPIEXEC} -n 1 ./ex30 -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type lu -pc_factor_shift_type INBLOCKS > ex30.tmp 2>&1; \
	   if (${DIFF} output/ex30_shiftinblocks.out ex30.tmp) then true; \
	   else echo "Possible problem with ex30_shiftinblocks_lu, diffs above"; fi; \
	   ${RM} -f ex30.tmp
	-@${MPIEXEC} -n 1 ./ex30 -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type cholesky -pc_factor_shift_type INBLOCKS > ex30.tmp 2>&1; \
	   if (${DIFF} output/ex30_shiftinblocks.out ex30.tmp) then true; \
	   else echo "Possible problem with ex30_shiftinblocks_cholesky, diffs above"; fi; \
	   ${RM} -f ex30.tmp

runex30_shift_in_blocks_sbaij: # sensitive to machine, thus not submitted to the nightly tests
	-@${MPIEXEC} -n 1 ./ex30 -f0 ${DATAFILESPATH}/matrices/small -mat_sigma -4.0 -ksp_type preonly -pc_type cholesky -pc_factor_shift_type INBLOCKS -mat_type sbaij > ex30.tmp 2>&1; \
	   if (${DIFF} output/ex30_shiftinblocks.out ex30.tmp) then true; \
	   else echo "Possible problem with ex30_shift_in_blocks_sbaij, diffs above"; fi; \
	   ${RM} -f ex30.tmp

runex31:
	-@${MPIEXEC} -n 3 ./ex31 -f ${DATAFILESPATH}/matrices/small -partition > ex31.tmp 2>&1; \
	if (${DIFF} output/ex31.out ex31.tmp) then true; \
	   else echo "Possible problem with ex31, diffs above"; fi; \
	   ${RM} -f ex31.tmp

runex32_testset1:
	@-count=0; \
	for i in 1 2 3 4 5; do \
	    for ordering in natural nd; do \
		for simtype in '' -trans;do \
		    for flevels in 0 1 2 3; do \
	    	    	count=`expr $$count + 1`; \
	   		echo "[$${count}]_start" > ex32.tmp 2>&1; \
		    	${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -dof $$i -pc_type ilu -pc_factor_mat_ordering_type $$ordering -mat_type aij $$simtype -pc_factor_levels $$flevels >> ex32.tmp 2>&1;  \
		    	echo "[$${count}]_end" >> ex32.tmp 2>&1;  \
		    	sed -n '/\['$${count}']_start/,/\['$${count}']_end/p' output/ex32_1.out > ex32.tmp2 2>&1;  \
		    	if (${DIFF} ex32.tmp ex32.tmp2) then true;  \
		    	    else echo "Possible problem with ./ex32 -ksp_monitor_short -dof $$i -pc_type ilu -pc_factor_mat_ordering_type $$ordering -mat_type aij $$simtype -pc_factor_levels $$flevels"; fi;   \
			    ${RM} -f ex32.tmp; ${RM} -f ex32.tmp2; \
	    	    done; \
		done; \
	    done; \
	done;

runex32_testset2:
	@-count=0; \
	for i in 1 2 3 4 5 6 7 8; do \
	    for ordering in natural nd; do \
		for simtype in '' -trans;do \
		    for flevels in 0 1 2 3; do \
                        count=`expr $$count + 1`; \
	   		echo "[$${count}]_start" > ex32.tmp 2>&1; \
		    	${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -dof $$i -pc_type ilu -pc_factor_mat_ordering_type $$ordering -mat_type baij $$simtype -pc_factor_levels $$flevels >> ex32.tmp 2>&1;  \
		    	    echo "[$${count}]_end" >> ex32.tmp 2>&1;  \
		    	    sed -n '/\['$${count}']_start/,/\['$${count}']_end/p' output/ex32_2.out > ex32.tmp2 2>&1;  \
		    	if (${DIFF} ex32.tmp ex32.tmp2) then true;  \
		    	    else echo "Possible problem with ./ex32 -ksp_monitor_short -dof $$i -pc_type ilu -pc_factor_mat_ordering_type $$ordering -mat_type baij $$simtype -pc_factor_levels $$flevels"; fi;   \
			${RM} -f ex32.tmp; ${RM} -f ex32.tmp2; \
	    	    done; \
		done; \
	    done; \
	done;

runex32_testset3:
	@-count=0; \
	for i in 1 2 3 4 5; do \
	    for ordering in natural nd; do \
		for simtype in '' -trans;do \
	    	    count=`expr $$count + 1`; \
	   	    echo "[$${count}]_start" > ex32.tmp 2>&1; \
		    ${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -dof $$i -pc_type lu -pc_factor_mat_ordering_type $$ordering -mat_type aij $$simtype >> ex32.tmp 2>&1;  \
		    echo "[$${count}]_end" >> ex32.tmp 2>&1;  \
		    sed -n '/\['$${count}']_start/,/\['$${count}']_end/p' output/ex32_3.out > ex32.tmp2 2>&1;  \
		    if (${DIFF} ex32.tmp ex32.tmp2) then true;  \
		    	else echo "Possible problem with ./ex32 -ksp_monitor_short -dof $$i -pc_type lu -pc_factor_mat_ordering_type $$ordering -mat_type aij $$simtype "; fi;   \
		    ${RM} -f ex32.tmp; ${RM} -f ex32.tmp2; \
		done; \
	    done; \
	done;

runex32_testset4:
	@-count=0; \
	for i in 1 2 3 4 5 6 7 8; do \
	    for ordering in natural nd; do \
		for simtype in '' -trans;do \
	    	    count=`expr $$count + 1`; \
	   	    echo "[$${count}]_start" > ex32.tmp 2>&1; \
		    ${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -dof $$i -pc_type lu -pc_factor_mat_ordering_type $$ordering -mat_type baij $$simtype >> ex32.tmp 2>&1;  \
		    echo "[$${count}]_end" >> ex32.tmp 2>&1;  \
		    sed -n '/\['$${count}']_start/,/\['$${count}']_end/p' output/ex32_4.out > ex32.tmp2 2>&1;  \
		    if (${DIFF} ex32.tmp ex32.tmp2) then true;  \
		    	else echo "Possible problem with ./ex32 -ksp_monitor_short -dof $$i -pc_type lu -pc_factor_mat_ordering_type $$ordering -mat_type baij $$simtype "; fi;   \
		    ${RM} -f ex32.tmp; ${RM} -f ex32.tmp2; \
		done; \
	    done; \
	done;

runex32_testset5:
	@-count=0; \
	for ordering in natural nd; do \
	    for flevels in 0 1 2 3; do \
		count=`expr $$count + 1`; \
	   	echo "[$${count}]_start" > ex32.tmp 2>&1; \
		${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -dof 1 -pc_type icc -ksp_type cg -pc_factor_mat_ordering_type $$ordering -mat_type aij -pc_factor_levels $$flevels >> ex32.tmp 2>&1;  \
		echo "[$${count}]_end" >> ex32.tmp 2>&1;  \
		sed -n '/\['$${count}']_start/,/\['$${count}']_end/p' output/ex32_5.out > ex32.tmp2 2>&1;  \
		if (${DIFF} ex32.tmp ex32.tmp2) then true;  \
		    	else echo "Possible problem with ./ex32 -ksp_monitor_short -dof 1 -pc_type icc -ksp_type cg -pc_factor_mat_ordering_type $$ordering -mat_type aij -pc_factor_levels $$flevels"; fi;   \
		${RM} -f ex32.tmp; ${RM} -f ex32.tmp2; \
	   done; \
	done; \
       
	@-count=8; \
        for ordering in natural nd; do \
	    count=`expr $$count + 1`; \
	    echo "[$${count}]_start" > ex32.tmp 2>&1; \
	    ${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -dof 1 -pc_type cholesky -ksp_type cg -pc_factor_mat_ordering_type $$ordering -mat_type aij >> ex32.tmp 2>&1;  \
	    echo "[$${count}]_end" >> ex32.tmp 2>&1;  \
	    sed -n '/\['$${count}']_start/,/\['$${count}']_end/p' output/ex32_5.out > ex32.tmp2 2>&1;  \
	    if (${DIFF} ex32.tmp ex32.tmp2) then true;  \
	    else echo "Possible problem with ./ex32 -ksp_monitor_short -dof 1 -pc_type cholesky -ksp_type cg -pc_factor_mat_ordering_type $$ordering -mat_type aij "; fi;   \
	    ${RM} -f ex32.tmp; ${RM} -f ex32.tmp2; \
	done; \
       

	@-count=10; \
	count=`expr $$count + 1`; \
	echo "[$${count}]_start" > ex32.tmp 2>&1; \
	${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -dof 1 -pc_type icc -test_sbaij1 >> ex32.tmp 2>&1;  \
	echo "[$${count}]_end" >> ex32.tmp 2>&1;  \
	sed -n '/\['$${count}']_start/,/\['$${count}']_end/p' output/ex32_5.out > ex32.tmp2 2>&1;  \
	if (${DIFF} ex32.tmp ex32.tmp2) then true;  \
	else echo "Possible problem with ./ex32 -ksp_monitor_short -dof 1 -pc_type icc -test_sbaij1"; fi;   \
	${RM} -f ex32.tmp; ${RM} -f ex32.tmp2

runex32_aij_sbaij_icc:
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -dof 1 -test_sbaij1 -mat_ignore_lower_triangular -pc_type icc > ex32_aij_sbaij_icc.out 2>&1;
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -dof 1 -mat_type aij -pc_type icc -pc_factor_mat_ordering_type natural > ex32.tmp 2>&1; \
	if (${DIFF} ex32_aij_sbaij_icc.out ex32.tmp) then true; \
	   else echo "Possible problem with ex32_aij_sbaij_icc, diffs above"; fi; \
	   ${RM} -f ex32.tmp; ${RM} -f ex32_aij_sbaij_icc.out

runex32_aij_sbaij_icc_lev1:
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -dof 1 -test_sbaij1 -ksp_type cg -pc_type icc -pc_factor_levels 1 > ex32_aij_sbaij_icc.out 2>&1;
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -dof 1 -mat_type aij -ksp_type cg -pc_type icc -pc_factor_mat_ordering_type natural -pc_factor_levels 1 > ex32.tmp 2>&1; \
	if (${DIFF} ex32_aij_sbaij_icc.out ex32.tmp) then true; \
	   else echo "Possible problem with ex32_aij_sbaij_icc_lev1, diffs above"; fi; \
	   ${RM} -f ex32.tmp; ${RM} -f ex32_aij_sbaij_icc.out

runex32_aij_sbaij_icc_lev10:
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -dof 1 -test_sbaij1 -ksp_type cg -pc_type icc -pc_factor_levels 10 > ex32_aij_sbaij_icc.out 2>&1;
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -dof 1 -mat_type aij -ksp_type cg -pc_type icc -pc_factor_mat_ordering_type natural -pc_factor_levels 10 > ex32.tmp 2>&1; \
	if (${DIFF} ex32_aij_sbaij_icc.out ex32.tmp) then true; \
	   else echo "Possible problem with ex32_aij_sbaij_icc_lev10, diffs above"; fi; \
	   ${RM} -f ex32.tmp; ${RM} -f ex32_aij_sbaij_icc.out

runex32_aij_sbaij_cholesky:
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -dof 1 -test_sbaij1 -pc_type cholesky > ex32_aij_sbaij_icc.out 2>&1;
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -dof 1 -mat_type aij -pc_type cholesky -pc_factor_mat_ordering_type natural > ex32.tmp 2>&1; \
	if (${DIFF} ex32_aij_sbaij_icc.out ex32.tmp) then true; \
	   else echo "Possible problem with ex32_aij_sbaij_icc_lev1, diffs above"; fi; \
	   ${RM} -f ex32.tmp; ${RM} -f ex32_aij_sbaij_icc.out

runex32_inode2:
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -mat_type aij -dof 2 > ex32_inode.out 2>&1;
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -mat_type aij -dof 2 -mat_no_inode > ex32.tmp 2>&1; \
	if (${DIFF} ex32_inode.out ex32.tmp) then true; \
	   else echo "Possible problem with ex32_inode2, diffs above"; fi; \
	   ${RM} -f ex32.tmp; ${RM} -f ex32_inode.out
runex32_inode2_nd:
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -mat_type aij -dof 2 -pc_factor_mat_ordering_type nd > ex32_inode.out 2>&1;
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -mat_type aij -dof 2 -pc_factor_mat_ordering_type nd -mat_no_inode > ex32.tmp 2>&1; \
	if (${DIFF} ex32_inode.out ex32.tmp) then true; \
	   else echo "Possible problem with ex32_inode2_nd, diffs above"; fi; \
	   ${RM} -f ex32.tmp; ${RM} -f ex32_inode.out

runex32_inode3:
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -mat_type aij -dof 3 > ex32_inode.out 2>&1;
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -mat_type aij -dof 3 -mat_no_inode > ex32.tmp 2>&1; \
	if (${DIFF} ex32_inode.out ex32.tmp) then true; \
	   else echo "Possible problem with ex32_inode3, diffs above"; fi; \
	   ${RM} -f ex32.tmp; ${RM} -f ex32_inode.out
runex32_inode3_nd:
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -mat_type aij -dof 3 -pc_factor_mat_ordering_type nd > ex32_inode.out 2>&1;
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -mat_type aij -dof 3 -pc_factor_mat_ordering_type nd -mat_no_inode > ex32.tmp 2>&1; \
	if (${DIFF} ex32_inode.out ex32.tmp) then true; \
	   else echo "Possible problem with ex32_inode3_nd, diffs above"; fi; \
	   ${RM} -f ex32.tmp; ${RM} -f ex32_inode.out

runex32_inode4:
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -mat_type aij -dof 4 > ex32_inode.out 2>&1;
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -mat_type aij -dof 4 -mat_no_inode > ex32.tmp 2>&1; \
	if (${DIFF} ex32_inode.out ex32.tmp) then true; \
	   else echo "Possible problem with ex32_inode4, diffs above"; fi; \
	   ${RM} -f ex32.tmp; ${RM} -f ex32_inode.out
runex32_inode4_nd:
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -mat_type aij -dof 4 -pc_factor_mat_ordering_type nd > ex32_inode.out 2>&1;
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -mat_type aij -dof 4 -pc_factor_mat_ordering_type nd -mat_no_inode > ex32.tmp 2>&1; \
	if (${DIFF} ex32_inode.out ex32.tmp) then true; \
	   else echo "Possible problem with ex32_inode4_nd, diffs above"; fi; \
	   ${RM} -f ex32.tmp; ${RM} -f ex32_inode.out

runex32_inode5:
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -mat_type aij -dof 5 > ex32_inode.out 2>&1;
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -mat_type aij -dof 5 -mat_no_inode > ex32.tmp 2>&1; \
	if (${DIFF} ex32_inode.out ex32.tmp) then true; \
	   else echo "Possible problem with ex32_inode5, diffs above"; fi; \
	   ${RM} -f ex32.tmp; ${RM} -f ex32_inode.out
runex32_inode5_nd:
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -mat_type aij -dof 5 -pc_factor_mat_ordering_type nd > ex32_inode.out 2>&1;
	-@${MPIEXEC} -n 1 ./ex32 -ksp_monitor_short -mat_type aij -dof 5 -pc_factor_mat_ordering_type nd -mat_no_inode > ex32.tmp 2>&1; \
	if (${DIFF} ex32_inode.out ex32.tmp) then true; \
	   else echo "Possible problem with ex32_inode5_nd, diffs above"; fi; \
	   ${RM} -f ex32.tmp; ${RM} -f ex32_inode.out

runex33:
	-@${MPIEXEC} -n 1 ./ex33 -sigma 2.0 > ex33.tmp 2>&1; \
	if (${DIFF} output/ex33.out ex33.tmp) then true; \
	   else echo "Possible problem with ex33, diffs above"; fi; \
	   ${RM} -f ex33.tmp
runex33_mumps:
	-@${MPIEXEC} -n 1 ./ex33 -sigma 2.0 -pc_factor_mat_solver_package mumps -mat_mumps_icntl_13 1 > ex33.tmp 2>&1; \
	if (${DIFF} output/ex33.out ex33.tmp) then true; \
	   else echo "Possible problem with ex33_mumps, diffs above"; fi; \
	   ${RM} -f ex33.tmp
runex33_mumps_2:
	-@${MPIEXEC} -n 3 ./ex33 -sigma 2.0 -pc_factor_mat_solver_package mumps -mat_mumps_icntl_13 1 > ex33.tmp 2>&1; \
	if (${DIFF} output/ex33.out ex33.tmp) then true; \
	   else echo "Possible problem with ex33_mumps_2, diffs above"; fi; \
	   ${RM} -f ex33.tmp
runex34:
	-@${MPIEXEC} -n 2 ./ex34 -f  ${DATAFILESPATH}/matrices/small -nodesize 2 > ex34.tmp 2>&1; \
	if (${DIFF} output/ex34_1.out ex34.tmp) then true; \
	   else echo "Possible problem with ex34, diffs above"; fi; \
	   ${RM} -f ex34.tmp

TESTEXAMPLES_C		       = ex1.PETSc ex1.rm ex3.PETSc runex3 runex3_2 ex3.rm ex4.PETSc runex4 runex4_3 \
                                 runex4_5 ex4.rm ex7.PETSc ex7.rm ex19.PETSc runex19 runex19_2 ex19.rm \
                                 ex24.PETSc runex24 runex24_2 runex24_3 runex24_4 ex24.rm \
                                 ex26.PETSc runex26 runex26_2 ex26.rm ex27.PETSc ex27.rm \
                                 ex32.PETSc runex32_testset1 runex32_testset2 runex32_testset3 runex32_testset4 runex32_testset5 \
                                 runex32_aij_sbaij_icc runex32_aij_sbaij_icc_lev1 runex32_aij_sbaij_icc_lev10 runex32_aij_sbaij_cholesky \
                                 runex32_inode2 runex32_inode2_nd runex32_inode3 runex32_inode3_nd runex32_inode4 runex32_inode4_nd \
                                 runex32_inode5 runex32_inode5_nd ex32.rm  
TESTEXAMPLES_C_X11	       = ex10.PETSc runex10 ex10.rm ex15.PETSc ex15.rm 
TESTEXAMPLES_C_NOCOMPLEX       = ex33.PETSc runex33 ex33.rm 
TESTEXAMPLES_FORTRAN	       = ex5f.PETSc runex5f ex5f.rm
TESTEXAMPLES_FORTRAN_MPIUNI    = ex12f.PETSc ex12f.rm
TESTEXAMPLES_C_X11_MPIUNI      = ex3.PETSc runex3 ex3.rm ex4.PETSc runex4 ex4.rm
TESTEXAMPLES_C_COMPLEX	       = ex17.PETSc runex17 ex17.rm
TESTEXAMPLES_C_COMPLEX_MPIUNI  = ex17.PETSc runex17 ex17.rm
TESTEXAMPLES_DATAFILESPATH     = ex6.PETSc runex6 runex6_2 runex6_3 ex6.r ex25.PETSc runex25 runex25_2 ex25.rm ex27.PETSc runex27 ex25.rm \
                                 ex30.PETSc runex30_shift runex30_shift_cholesky_aij runex30_shift_cholesky_sbaij ex30.rm \
                                 ex31.PETSc runex31 ex31.rm \
                                 ex34.PETSc runex34 ex34.rm
TESTEXAMPLES_FORTRAN_NOCOMPLEX = ex16f.PETSc runex16f ex16f.rm  ex12f.PETSc runex12f ex12f.rm 
TESTEXAMPLES_13		       = ex18.PETSc ex18.rm ex20.PETSc runex20 ex20.rm ex5f.PETSc ex5f.rm
TESTEXAMPLES_ML                = ex26.PETSc runex26_ml runex26_ml_2 runex26_ml_3 ex26.rm ex29.PETSc runex29 runex29_2 ex29.rm

include ${PETSC_DIR}/conf/test
