# $Id: makefile,v 1.65 1999/11/05 14:46:58 bsmith Exp bsmith $ 

ALL: ex1

CFLAGS	         =
FFLAGS	         = 
CPPFLAGS         = ${BS_INCLUDE}
FPPFLAGS         =
LOCDIR           = src/sles/examples/tutorials/

TESTEXAMPLES_1   = ex1.PETSc runex1 runex1_2 runex1_3 \
		   ex1.rm ex2.PETSc runex2 runex2_2 runex2_3 runex2_4 \
		   ex2.rm ex7.PETSc runex7 ex7.rm ex5.PETSc runex5 \
		   runex5_2 ex5.rm ex9.PETSc runex9 ex9.rm \
                   ex12.PETSc runex12 ex12.rm ex13.PETSc runex13 ex13.rm \
                   ex14.PETSc runex14 runex14_2 ex14.rm \
                   ex15.PETSc runex15 ex15.rm \
                   ex16.PETSc runex16 ex16.rm ex18.PETSc runex18 ex18.rm clean	
TESTEXAMPLES_2   = ex2.PETSc runex2_5 ex2.rm ex4.PETSc ex4.rm ex8.PETSc ex5.PETSc \
	           runex5_5 ex5.rm ex8.rm ex10.PETSc runex10_7 runex10_8 runex10_9 ex10.rm clean
TESTEXAMPLES_3   = ex1f.PETSc runex1f ex1f.rm ex2f.PETSc runex2f ex2f.rm \
                   ex6f.PETSc ex6f.rm ex14f.PETSc runex14f ex14f.rm ex15f.PETSc runex15f ex15f.rm clean
TESTEXAMPLES_9   = ex1f.PETSc runex1f ex1f.rm ex6f.PETSc runex6f ex6f.rm clean
TESTEXAMPLES_4   = ex1.PETSc runex1 runex1_2 runex1_3 ex1.rm ex2.PETSc runex2 \
		   runex2_3 ex2.rm ex7.PETSc ex7.rm ex5.PETSc ex5.rm \
		   ex4.PETSc ex4.rm ex9.PETSc runex9 ex9.rm clean
TESTEXAMPLES_5   = ex10.PETSc runex10 ex10.rm ex11.PETSc runex11 ex11.rm clean
TESTEXAMPLES_5   = ex11.PETSc runex11 ex11.rm clean
TESTEXAMPLES_6   = ex10.PETSc runex10_2 runex10_3 runex10_4 \
                   runex10_5 runex10_6 runex10_8 ex10.rm 
TESTEXAMPLES_7   = ex5.PETSc runex5_3 runex5_4 ex5.rm clean
TESTEXAMPLES_8   = 
TESTEXAMPLES_10  = ex11f.PETSc runex11f ex11f.rm clean
TESTEXAMPLES_12  = ex13f90.PETSc runex13f90 ex13f90.rm clean
TESTEXAMPLES_13  = ex3.PETSc ex3.rm ex17.PETSc ex17.rm ex14f.PETSc ex14f.rm clean
BUILDEXAMPLES_1  = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex7.PETSc ex7.rm \
		   ex5.PETSc ex5.rm ex9.PETSc ex9.rm \
		   ex13.PETSc ex13.rm ex14.PETSc ex14.rm \
                   ex15.PETSc ex15.rm ex16.PETSc ex16.rm clean
BUILDEXAMPLES_2  = ex4.PETSc ex4.rm ex8.PETSc ex8.rm clean
BUILDEXAMPLES_3  = ex1f.PETSc ex1f.rm ex2f.PETSc ex2f.rm ex6f.PETSc ex6f.rm \
                   ex14f.PETSc ex14f.rm ex15f.PETSc ex15f.rm clean
BUILDEXAMPLES_4  = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex7.PETSc ex7.rm \
		   ex5.PETSc ex5.rm ex4.PETSc ex4.rm ex9.PETSc ex9.rm clean
BUILDEXAMPLES_5  = ex11.PETSc ex11.rm clean
BUILDEXAMPLES_11 = ex11.PETSc ex11.rm clean
BUILDEXAMPLES_6  = ex5.PETSc ex5.rm ex10.PETSc ex10.rm clean
BUILDEXAMPLES_7  = ex5.PETSc ex5.rm clean
BUILDEXAMPLES_8  = 
BUILDEXAMPLES_9  = ex1f.PETSc ex1f.rm clean
BUILDEXAMPLES_10 = ex11f.PETSc ex11f.rm clean

EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex7.c ex8.c ex9.c ex10.c ex11.c \
                  ex12.c ex13.c ex14.c ex15.c ex16.c ex17.c ex18.c
EXAMPLESF       = ex1f.F ex2f.F ex6f.F ex11f.F ex13f90.F ex14f.F ex15f.F

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex0: ex0.o  chkopts
	-${CLINKER} -o ex0 ex0.o  ${PETSC_SLES_LIB}
	${RM} ex0.o

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_SLES_LIB}
	${RM} ex1.o

ex0f: ex0f.o  chkopts
	-${FLINKER} -o ex0f ex0f.o  ${PETSC_FORTRAN_LIB} ${PETSC_SLES_LIB}
	${RM} ex0f.o

ex1f: ex1f.o  chkopts
	-${FLINKER} -o ex1f ex1f.o  ${PETSC_FORTRAN_LIB} ${PETSC_SLES_LIB}
	${RM} ex1f.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_SLES_LIB}
	${RM} ex2.o

ex2a: ex2a.o  chkopts
	-${CLINKER} -o ex2a ex2a.o  ${PETSC_SLES_LIB}
	${RM} ex2a.o

ex2f: ex2f.o  chkopts
	-${FLINKER} -o ex2f ex2f.o  ${PETSC_FORTRAN_LIB} ${PETSC_SLES_LIB}
	${RM} ex2f.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_SLES_LIB}
	${RM} ex3.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_SLES_LIB}
	${RM} ex4.o

ex5: ex5.o  chkopts
	-${CLINKER} -o ex5 ex5.o  ${PETSC_SLES_LIB}
	${RM} ex5.o

ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_SLES_LIB}
	${RM} ex6.o

ex6f: ex6f.o  chkopts
	-${FLINKER} -o ex6f ex6f.o  ${PETSC_FORTRAN_LIB} ${PETSC_SLES_LIB}
	${RM} ex6f.o

ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_SLES_LIB}
	${RM} ex7.o

ex8: ex8.o  chkopts
	-${CLINKER} -o ex8 ex8.o  ${PETSC_SLES_LIB}
	${RM} ex8.o

ex9: ex9.o  chkopts
	-${CLINKER} -o ex9 ex9.o  ${PETSC_SLES_LIB}
	${RM} ex9.o

ex10: ex10.o  chkopts
	-${CLINKER} -o ex10 ex10.o  ${PETSC_SLES_LIB}
	${RM} ex10.o

ex11: ex11.o  chkopts
	-${CLINKER} -o ex11 ex11.o  ${PETSC_SLES_LIB}
	${RM} ex11.o

ex11f: ex11f.o  chkopts
	-${FLINKER} -o ex11f ex11f.o ${PETSC_FORTRAN_LIB} ${PETSC_SLES_LIB}
	${RM} ex11f.o

ex12: ex12.o  chkopts
	-${CLINKER} -o ex12 ex12.o  ${PETSC_SLES_LIB}
	${RM} ex12.o

ex13: ex13.o  chkopts
	-${CLINKER} -o ex13 ex13.o  ${PETSC_SLES_LIB}
	${RM} ex13.o

ex13f90: ex13f90.o  chkopts
	-${FLINKER} -o ex13f90 ex13f90.o ${PETSC_FORTRAN_LIB} ${PETSC_SLES_LIB}
	${RM} ex13f90.o

ex14: ex14.o  chkopts
	-${CLINKER} -o ex14 ex14.o  ${PETSC_SLES_LIB}
	${RM} ex14.o

ex14f: ex14f.o  chkopts
	-${FLINKER} -o ex14f ex14f.o  ${PETSC_FORTRAN_LIB} ${PETSC_SLES_LIB}
	${RM} ex14f.o

ex15: ex15.o  chkopts
	-${CLINKER} -o ex15 ex15.o  ${PETSC_SLES_LIB}
	${RM} ex15.o

ex15f: ex15f.o  chkopts
	-${FLINKER} -o ex15f ex15f.o  ${PETSC_FORTRAN_LIB} ${PETSC_SLES_LIB}
	${RM} ex15f.o

ex16: ex16.o  chkopts
	-${CLINKER} -o ex16 ex16.o  ${PETSC_SLES_LIB}
	${RM} ex16.o

ex17: ex17.o  chkopts
	-${CLINKER} -o ex17 ex17.o  ${PETSC_SLES_LIB}
	${RM} ex17.o

ex18: ex18.o  chkopts
	-${CLINKER} -o ex18 ex18.o  ${PETSC_SLES_LIB}
	${RM} ex18.o

runex1:
	-@${MPIRUN} -np 1 ex1 -ksp_smonitor > ex1_1.tmp 2>&1;	  \
	   if (diff output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   ${RM} -f ex1_1.tmp
runex1_2:
	-@${MPIRUN} -np 1 ex1 -pc_type sor -pc_sor_symmetric -ksp_smonitor >\
	   ex1_2.tmp 2>&1;   \
	   if (diff output/ex1_2.out ex1_2.tmp) then true; \
	   else echo "Possible problem with ex1_2, diffs above"; fi; \
	   ${RM} -f ex1_2.tmp
runex1_3:
	-@${MPIRUN} -np 1 ex1 -pc_type eisenstat -ksp_smonitor >\
	   ex1_3.tmp 2>&1;   \
	   if (diff output/ex1_3.out ex1_3.tmp) then true; \
	   else echo "Possible problem with ex1_3, diffs above"; fi; \
	   ${RM} -f ex1_3.tmp
runex1f:
	-@${MPIRUN} -np 1 ex1f -ksp_smonitor > ex1f_1.tmp 2>&1;	  \
	   if (diff output/ex1f_1.out ex1f_1.tmp) then true; \
	   else echo "Possible problem with ex1f_1, diffs above"; fi; \
	   ${RM} -f ex1f_1.tmp
runex2:
	-@${MPIRUN} -np 1 ex2 -ksp_smonitor -m 5 -n 5 > ex2_1.tmp 2>&1; \
	   if (diff output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi; \
	   ${RM} -f ex2_1.tmp
runex2_2:
	-@${MPIRUN} -np 2 ex2 -ksp_smonitor -m 5 -n 5 > ex2_2.tmp 2>&1; \
	   if (diff output/ex2_2.out ex2_2.tmp) then true; \
	   else echo "Possible problem with ex2_2, diffs above"; fi; \
	   ${RM} -f ex2_2.tmp
runex2_3:
	-@${MPIRUN} -np 1 ex2 -pc_type sor -pc_sor_symmetric -ksp_smonitor > \
	    ex2_3.tmp 2>&1;   \
	   if (diff output/ex2_3.out ex2_3.tmp) then true; \
	   else echo "Possible problem with ex2_3, diffs above"; fi; \
	   ${RM} -f ex2_3.tmp
runex2_4:
	-@${MPIRUN} -np 1 ex2 -pc_type eisenstat -ksp_smonitor >\
	    ex2_4.tmp 2>&1;   \
	   if (diff output/ex2_4.out ex2_4.tmp) then true; \
	   else echo "Possible problem with ex2_4, diffs above"; fi; \
	   ${RM} -f ex2_4.tmp
runex2_5:
	-@${MPIRUN} -np 2 ex2 -ksp_smonitor -m 5 -n 5 -mat_view_draw > ex2_5.tmp 2>&1; \
	   if (diff output/ex2_2.out ex2_5.tmp) then true; \
	   else echo "Possible problem with ex2_5, diffs above"; fi; \
	   ${RM} -f ex2_4.tmp
runex2f:
	-@${MPIRUN} -np 2 ex2f -pc_type jacobi -ksp_smonitor > ex2f_1.tmp 2>&1; \
	   if (diff output/ex2f_1.out ex2f_1.tmp) then true; \
	   else echo "Possible problem with ex2f_1, diffs above"; fi; \
	   ${RM} -f ex2f_1.tmp
runex4:
	-@${MPIRUN} -np 3  ex4 -ksp_smonitor > ex4_1.tmp 2>&1; \
	   if (diff output/ex4_1.out ex4_1.tmp) then true; \
	   else echo "Possible problem with ex4_1, diffs above"; fi; \
	   ${RM} -f ex4_1.tmp
runex5:
	-@${MPIRUN} -np 1 ex5 -pc_type jacobi -ksp_smonitor > ex5_1.tmp 2>&1; \
	   if (diff output/ex5_1.out ex5_1.tmp) then true; \
	   else echo "Possible problem with ex5_1, diffs above"; fi; \
	   ${RM} -f ex5_1.tmp
runex5_2:
	-@${MPIRUN} -np 2 ex5 -pc_type jacobi -ksp_smonitor \
	   -ksp_rtol .000001 > ex5_2.tmp 2>&1;   \
	   if (diff output/ex5_2.out ex5_2.tmp) then true; \
	   else echo "Possible problem with ex5_2, diffs above"; fi; \
	   ${RM} -f ex5_2.tmp

# Note: We test these BlockSolve95 examples differently from the usual
# because the coloring heuristic is machine-dependent; convergence rates
# may differ slightly on various architectures.
runex5_3:
	-@${MPIRUN} -np 2 ex5 -pc_type icc -ksp_type cg -mat_mpirowbs -ksp_rtol .000001;\
	echo "ex5_3: This program tested the ICC preconditioner in BlockSolve95."
runex5_4:
	-@${MPIRUN} -np 2 ex5 -pc_type ilu -mat_nonsym -mat_mpirowbs \
	   -ksp_rtol .000001;\
	echo "ex5_4: This program tested the ILU preconditioner in BlockSolve95."
runex5_5:
	-@${MPIRUN} -np 2 ex5 -mat_bdiag -mat_block_size 2 > ex5_5.tmp 2>&1; \
	   if (diff output/ex5_5.out ex5_5.tmp) then true; \
	   else echo "Possible problem with ex5_5, diffs above"; fi; \
	   ${RM} -f ex5_5.tmp
runex6f:
	-@${MPIRUN} -np 1 ex6f -pc_type jacobi -mat_view -ksp_smonitor > ex6f_1.tmp 2>&1; \
	   if (diff output/ex6f_1.out ex6f_1.tmp) then true; \
	   else echo "Possible problem with ex6f_1, diffs above"; fi; \
	   ${RM} -f ex6f_1.tmp
runex7:
	-@${MPIRUN} -np 2 ex7 -ksp_smonitor -noslesview> ex7_1.tmp 2>&1; \
	   if (diff output/ex7_1.out ex7_1.tmp) then true; \
	   else echo "Possible problem with ex7_1, diffs above"; fi; \
	   ${RM} -f ex7_1.tmp
runex9:
	-@${MPIRUN} -np 1 ex9 -t 2 -pc_type jacobi -ksp_smonitor -ksp_type gmres \
	   -s2_ksp_type bcgs -s2_pc_type jacobi -s2_ksp_smonitor \
           > ex9_1.tmp 2>&1; \
	   if (diff output/ex9_1.out ex9_1.tmp) then true; \
	   else echo "Possible problem with ex9_1, diffs above"; fi; \
	   ${RM} -f ex9_1.tmp
runex10:
	-@${MPIRUN} -np 2 ex10 -f0 ../../../mat/examples/matbinary.cplx.ex > ex10_1.tmp 2>&1; \
	   if (diff output/ex10_1.out ex10_1.tmp) then true; \
	   else echo "Possible problem with ex10_1, diffs above"; fi; \
	   ${RM} -f ex10_1.tmp
runex10_2:
	-@${MPIRUN} -np 2 ex10 -options_left_off -ksp_type bicg \
	   -f0 ../../../mat/examples/matrices/medium > ex10_2.tmp 2>&1; \
	   if (diff output/ex10_2.out ex10_2.tmp) then true; \
	   else echo "Possible problem with ex10_2, diffs above"; fi; \
	   ${RM} -f ex10_2.tmp
runex10_3:
	-@${MPIRUN} -np 2 ex10 -options_left_off -ksp_type bicg -pc_type asm \
	   -f0 ../../../mat/examples/matrices/medium > ex10_3.tmp 2>&1; \
	   if (diff output/ex10_3.out ex10_3.tmp) then true; \
	   else echo "Possible problem with ex10_3, diffs above"; fi; \
	   ${RM} -f ex10_3.tmp
runex10_4:
	-@${MPIRUN} -np 1 ex10 -options_left_off -ksp_type bicg -pc_type lu \
	   -f0 ../../../mat/examples/matrices/medium > ex10_4.tmp 2>&1; \
	   if (diff output/ex10_4.out ex10_4.tmp) then true; \
	   else echo "Possible problem with ex10_4, diffs above"; fi; \
	   ${RM} -f ex10_4.tmp
runex10_5:
	-@${MPIRUN} -np 1 ex10 -options_left_off -ksp_type bicg \
	   -f0 ../../../mat/examples/matrices/medium > ex10_5.tmp 2>&1; \
	   if (diff output/ex10_5.out ex10_5.tmp) then true; \
	   else echo "Possible problem with ex10_5, diffs above"; fi; \
	   ${RM} -f ex10_5.tmp
runex10_6:
	-@${MPIRUN} -np 1 ex10 -pc_ilu_levels 2 -pc_ilu_fill 1.73 \
	   -f0 ../../../mat/examples/matrices/fem1 > ex10_6.tmp 2>&1; \
	   if (diff output/ex10_6.out ex10_6.tmp) then true; \
	   else echo "Possible problem with ex10_6, diffs above"; fi; \
	   ${RM} -f ex10_6.tmp
BS = 2 3 4 5 6 7 8
runex10_7:
	-@echo > ex10_7.tmp
	-@for bs in ${BS}; do \
	 ${MPIRUN} -np 1 ex10 -f0 ../../../mat/examples/matrices/medium -load_ignore_info \
                    -mat_seqbaij -matload_block_size $$bs -ksp_max_it 100 -ksp_rtol \
                    1.0e-15 -ksp_smonitor >> ex10_7.tmp 2>&1 ; \
	 ${MPIRUN} -np 1 ex10 -f0 ../../../mat/examples/matrices/medium -load_ignore_info \
                    -mat_seqbaij -matload_block_size $$bs -ksp_max_it 100 -ksp_rtol \
                    1.0e-15 -ksp_smonitor -mat_ordering_type nd >> ex10_7.tmp 2>&1 ; \
	 ${MPIRUN} -np 1 ex10 -f0 ../../../mat/examples/matrices/medium -load_ignore_info \
                    -mat_seqbaij -matload_block_size $$bs -ksp_max_it 100 -ksp_rtol \
                    1.0e-15 -ksp_smonitor -pc_ilu_levels 1 >> ex10_7.tmp 2>&1 ; \
	 ${MPIRUN} -np 1 ex10 -f0 ../../../mat/examples/matrices/medium -load_ignore_info \
                    -mat_seqbaij -matload_block_size $$bs -ksp_type preonly \
                    -ksp_smonitor -pc_type lu  >> ex10_7.tmp 2>&1 ; \
         done; cp ex10_7.tmp joe \
	-@if (diff output/ex10_7.out ex10_7.tmp) then true; \
	   else echo "Possible problem with ex10_7, diffs above"; fi;
	-@${RM} -f ex10_7.tmp
runex10_8:
	-@${MPIRUN} -np 1 ex10 -sles_diagonal_scale -pc_type eisenstat -ksp_smonitor -sles_diagonal_scale_fix \
	   -f0 ../../../mat/examples/matrices/medium -options_left_off > ex10_8.tmp 2>&1; \
	   if (diff output/ex10_8.out ex10_8.tmp) then true; \
	   else echo "Possible problem with ex10_8, diffs above"; fi; \
	   ${RM} -f ex10_8.tmp
runex10_9:
	-@echo > ex10_9.tmp
	-@for type in gmres bicg bcgs tcqmr tfqmr; do \
          for bs in 1 2 3 4 5 6 7; do \
          for np in 1 2 3; do \
	 ${MPIRUN} -np $$np ex10 -f0 ../../../mat/examples/matrices/medium -load_ignore_info \
                    -mat_baij -matload_block_size $$bs -ksp_max_it 100 -ksp_rtol \
                    1.0e-15 -ksp_smonitor >> ex10_9.tmp 2>&1 ; \
	 ${MPIRUN} -np $$np ex10 -f0 ../../../mat/examples/matrices/medium -load_ignore_info \
                    -mat_baij -matload_block_size $$bs -ksp_max_it 100 -ksp_rtol \
                    1.0e-15 -ksp_smonitor -trans >> ex10_9.tmp 2>&1 ; \
         done; done; done; 
	-@if (diff output/ex10_9.out ex10_9.tmp) then true; \
	   else echo "Possible problem with ex10_9, diffs above"; fi;
	-@${RM} -f ex10_9.tmp
runex11:
	-@${MPIRUN} -np 1 ex11 -n 6 -norandom -pc_type none -ksp_smonitor > ex11_1.tmp 2>&1; \
	   if (diff output/ex11_1.out ex11_1.tmp) then true; \
	   else echo "Possible problem with ex11_1, diffs above"; fi; \
	   ${RM} -f ex11_1.tmp
runex11f:
	-@${MPIRUN} -np 1 ex11f -n 6 -norandom -pc_type none -ksp_smonitor > ex11f_1.tmp 2>&1; \
	   if (diff output/ex11f_1.out ex11f_1.tmp) then true; \
	   else echo "Possible problem with ex11f_1, diffs above"; fi; \
	   ${RM} -f ex11f_1.tmp
runex12:
	-@${MPIRUN} -np 1 ex12  > ex12_1.tmp 2>&1; \
	   if (diff output/ex12_1.out ex12_1.tmp) then true; \
	   else echo "Possible problem with ex12_1, diffs above"; fi; \
	   ${RM} -f ex12_1.tmp
runex13:
	-@${MPIRUN} -np 1 ex13 -m 19 -n 20 > ex13_1.tmp 2>&1; \
	   if (diff output/ex13_1.out ex13_1.tmp) then true; \
	   else echo "Possible problem with ex13_1, diffs above"; fi; \
	   ${RM} -f ex13_1.tmp
runex13f90:
	-@${MPIRUN} -np 1 ex13f90 -m 19 -n 20 > ex13f90_1.tmp 2>&1; \
	   if (diff output/ex13f90_1.out ex13f90_1.tmp) then true; \
	   else echo "Possible problem with ex13f90_1, diffs above"; fi; \
	   ${RM} -f ex13f90_1.tmp
runex14:
	-@${MPIRUN} -np 1 ex14 -no_output > ex14_1.tmp 2>&1; \
	   if (diff output/ex14_1.out ex14_1.tmp) then true; \
	   else echo "Possible problem with ex14_1, diffs above"; fi; \
	   ${RM} -f ex14_1.tmp
runex14_2:
	-@${MPIRUN} -np 1 ex14 -no_output > ex14_2.tmp 2>&1; \
	   if (diff output/ex14_2.out ex14_2.tmp) then true; \
	   else echo "Possible problem with ex14_2, diffs above"; fi; \
	   ${RM} -f ex14_2.tmp
runex14f:
	-@${MPIRUN} -np 1 ex14f -no_output > ex14_1.tmp 2>&1; \
	   if (diff output/ex14_1.out ex14_1.tmp) then true; \
	   else echo "Possible problem with ex14f_1, diffs above"; fi; \
	   ${RM} -f ex14_1.tmp
runex15:
	-@${MPIRUN} -np 2 ex15 -sles_view -user_defined_pc > ex15_1.tmp 2>&1; \
	   if (diff output/ex15_1.out ex15_1.tmp) then true; \
	   else echo "Possible problem with ex15_1, diffs above"; fi; \
	   ${RM} -f ex15_1.tmp
runex15f:
	-@${MPIRUN} -np 2 ex15f -sles_view -user_defined_pc > ex15f_1.tmp 2>&1; \
	   if (diff output/ex15f_1.out ex15f_1.tmp) then true; \
	   else echo "Possible problem with ex15f_1, diffs above"; fi; \
	   ${RM} -f ex15f_1.tmp
runex16:
	-@${MPIRUN} -np 2 ex16 -ntimes 4 > ex16_1.tmp 2>&1; \
	   if (diff output/ex16_1.out ex16_1.tmp) then true; \
	   else echo "Possible problem with ex16_1, diffs above"; fi; \
	   ${RM} -f ex16_1.tmp
runex18:
	-@${MPIRUN} -np 2 ex18 -ksp_smonitor > ex18_1.tmp 2>&1; \
	   if (diff output/ex18_1.out ex18_1.tmp) then true; \
	   else echo "Possible problem with ex18_1, diffs above"; fi; \
	   ${RM} -f ex18_1.tmp


