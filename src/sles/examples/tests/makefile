# $Id: makefile,v 1.162 1999/02/01 21:30:13 balay Exp balay $ 

ALL: ex1

CFLAGS	  = 
FFLAGS    =
CPPFLAGS  = ${BS_INCLUDE}
FPPFLAGS  =
LOCDIR    = src/sles/examples/tests/
NOADIC    = true

TESTEXAMPLES_1  = ex1.PETSc ex1.rm \
                  ex2.PETSc runex2 ex2.rm ex3.PETSc runex3 runex3_2 ex3.rm ex4.PETSc\
		  runex4 runex4_3 runex4_5 ex4.rm ex7.PETSc ex7.rm clean
TESTEXAMPLES_2  = ex10.PETSc ex10.rm ex15.PETSc ex15.rm clean
TESTEXAMPLES_3  = ex12.PETSc ex12.rm clean
TESTEXAMPLES_9  = ex12.PETSc ex12.rm clean
TESTEXAMPLES_4  = ex3.PETSc runex3 ex3.rm ex4.PETSc \
		  runex4 ex4.rm clean
TESTEXAMPLES_5  = ex17.PETSc runex17 ex17.rm clean
TESTEXAMPLES_11 = ex17.PETSc runex17 ex17.rm clean
TESTEXAMPLES_6  = ex6.PETSc runex6 runex6_2 runex6_3 \
		  ex6.rm clean
TESTEXAMPLES_7  = ex6.PETSc runex6_6 ex6.rm clean
TESTEXAMPLES_8  = ex16.PETSc ex16.rm clean
TESTEXAMPLES_13 = ex18.PETSc ex18.rm ex5f.PETSc ex5f.rm clean
BUILDEXAMPLES_1 = ex2.PETSc ex2.rm ex3.PETSc ex3.rm ex4.PETSc ex4.rm clean
BUILDEXAMPLES_2 = ex10.PETSc ex10.rm ex15.PETSc ex15.rm clean
BUILDEXAMPLES_3 = ex12.PETSc ex12.rm ex16.PETSc ex16.rm clean
BUILDEXAMPLES_4 = ex3.PETSc ex3.rm ex4.PETSc ex4.rm clean
BUILDEXAMPLES_5 = ex17.PETSc ex17.rm clean
BUILDEXAMPLES_11= ex17.PETSc ex17.rm clean
BUILDEXAMPLES_6 = ex6.PETSc ex6.rm clean
BUILDEXAMPLES_7 = ex6.PETSc ex6.rm clean
BUILDEXAMPLES_8 = ex16.PETSc ex16.rm clean
BUILDEXAMPLES_9 = ex12.PETSc ex12.rm clean

EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex6.c ex7.c ex10.c \
		  ex15.c ex17.c ex18.c
EXAMPLESCH      = 
EXAMPLESF       = ex5f.F ex12.F ex16.F

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_SLES_LIB}
	${RM} ex1.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_SLES_LIB}
	${RM} ex2.o

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

ex12: ex12.o  chkopts
	-${FLINKER} -o ex12 ex12.o  ${PETSC_FORTRAN_LIB} ${PETSC_SLES_LIB}
	${RM} ex12.o

ex5f: ex5f.o  chkopts
	-${FLINKER} -o ex5f ex5f.o  ${PETSC_FORTRAN_LIB} ${PETSC_SLES_LIB}
	${RM} ex5f.o

ex13: ex13.o  chkopts
	-${CLINKER} -o ex13 ex13.o  ${PETSC_SLES_LIB}
	${RM} ex13.o

ex14: ex14.o  chkopts
	-${CLINKER} -o ex14 ex14.o  ${PETSC_SLES_LIB}
	${RM} ex14.o

ex15: ex15.o  chkopts
	-${CLINKER} -o ex15 ex15.o  ${PETSC_SLES_LIB}
	${RM} ex15.o

ex16: ex16.o  chkopts
	-${FLINKER} -o ex16 ex16.o  ${PETSC_FORTRAN_LIB} ${PETSC_SLES_LIB}
	${RM} ex16.o

ex17: ex17.o  chkopts
	-${CLINKER} -o ex17 ex17.o  ${PETSC_SLES_LIB}
	${RM} ex17.o

ex18: ex18.o  chkopts
	-${CLINKER} -o ex18 ex18.o  ${PETSC_SLES_LIB}
	${RM} ex18.o

ex19: ex19.o  chkopts
	-${CLINKER} -o ex19 ex19.o  ${PETSC_SLES_LIB}
	${RM} ex19.o

ex20: ex20.o  chkopts
	-${FLINKER} -o ex20 ex20.o  ${PETSC_FORTRAN_LIB} ${PETSC_SLES_LIB}
	${RM} ex20.o

ex21: ex21.o  chkopts
	-${CLINKER} -o ex21 ex21.o   ${PETSC_SLES_LIB}
	${RM} ex21.o

ex22: ex22.o  chkopts
	-${CLINKER} -o ex22 ex22.o  ${PETSC_SLES_LIB}
	${RM} ex22.o

runex1:
	-@${MPIRUN} -np 1 ex1 -pc_type jacobi -ksp_smonitor > ex1_1.tmp 2>&1;	  \
	   if (diff output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   ${RM} -f ex1_1.tmp
runex1_2:
	-@${MPIRUN} -np 2 ex1 -pc_type jacobi -ksp_smonitor > ex1_2.tmp 2>&1;	 \
	   if (diff output/ex1_2.out ex1_2.tmp) then true; \
	   else echo "Possible problem with ex1_2, diffs above"; fi; \
	   ${RM} -f ex1_2.tmp
runex1_3:
	-@${MPIRUN} -np 1 ex1 -pc_type sor -pc_sor_symmetric -ksp_smonitor >\
	   ex1_3.tmp 2>&1;   \
	   if (diff output/ex1_3.out ex1_3.tmp) then true; \
	   else echo "Possible problem with ex1_3, diffs above"; fi; \
	   ${RM} -f ex1_3.tmp
runex1_5:
	-@${MPIRUN} -np 1 ex1 -pc_type eisenstat -ksp_smonitor >\
	   ex1_5.tmp 2>&1;   \
	   if (diff output/ex1_5.out ex1_5.tmp) then true; \
	   else echo "Possible problem with ex1_5, diffs above"; fi; \
	   ${RM} -f ex1_5.tmp
runex2:
	-@${MPIRUN} -np 1 ex2 > ex2_1.tmp 2>&1; \
	   if (diff output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi; \
	   ${RM} -f ex2_1.tmp
runex2_2:
	-@${MPIRUN} -np 2 ex2 -pc_type jacobi -ksp_smonitor -m 5 > ex2_2.tmp 2>&1; \
	   if (diff output/ex2_2.out ex2_2.tmp) then true; \
	   else echo "Possible problem with ex2_2, diffs above"; fi; \
	   ${RM} -f ex2_2.tmp
runex2_3:
	-@${MPIRUN} -np 1 ex2 -pc_type sor -pc_sor_symmetric -ksp_smonitor > \
	    ex2_3.tmp 2>&1;   \
	   if (diff output/ex2_3.out ex2_3.tmp) then true; \
	   else echo "Possible problem with ex2_3, diffs above"; fi; \
	   ${RM} -f ex2_3.tmp
runex2_5:
	-@${MPIRUN} -np 1 ex2 -pc_type eisenstat -ksp_smonitor >\
	    ex2_5.tmp 2>&1;   \
	   if (diff output/ex2_5.out ex2_5.tmp) then true; \
	   else echo "Possible problem with ex2_5, diffs above"; fi; \
	   ${RM} -f ex2_5.tmp
runex3:
	-@${MPIRUN} -np 1 ex3 -pc_type jacobi -ksp_smonitor -m 5 > ex3_1.tmp 2>&1;   \
	   if (diff output/ex3_1.out ex3_1.tmp) then true; \
	   else echo "Possible problem with ex3_1, diffs above"; fi; \
	   ${RM} -f ex3_1.tmp
runex3_2:
	-@${MPIRUN} -np 2 ex3 -pc_type jacobi -ksp_smonitor -m 5 > ex3_2.tmp 2>&1;   \
	   if (diff output/ex3_2.out ex3_2.tmp) then true; \
	   else echo "Possible problem with ex3_2, diffs above"; fi; \
	   ${RM} -f ex3_2.tmp
runex4:
	-@${MPIRUN} -np 1 ex4 -ksp_smonitor -m 5 -pc_type jacobi > ex4_1.tmp 2>&1;\
	   if (diff output/ex4_1.out ex4_1.tmp) then true; \
	   else echo "Possible problem with ex4_1, diffs above"; fi; \
	   ${RM} -f ex4_1.tmp
runex4_3:
	-@${MPIRUN} -np 1 ex4 -pc_type sor -pc_sor_symmetric -ksp_smonitor -m 5 >\
	   ex4_3.tmp 2>&1;   \
	   if (diff output/ex4_3.out ex4_3.tmp) then true; \
	   else echo "Possible problem with ex4_3, diffs above"; fi; \
	   ${RM} -f ex4_3.tmp
runex4_5:
	-@${MPIRUN} -np 1 ex4 -pc_type eisenstat -ksp_smonitor -m 5	 >\
	   ex4_5.tmp 2>&1;   \
	   if (diff output/ex4_5.out ex4_5.tmp) then true; \
	   else echo "Possible problem with ex4_5, diffs above"; fi; \
	   ${RM} -f ex4_5.tmp
runex5:
	-@${MPIRUN} -np 2 ex5 -ksp_smonitor -noslesview> ex5_1.tmp 2>&1; \
	   if (diff output/ex5_1.out ex5_1.tmp) then true; \
	   else echo "Possible problem with ex5_1, diffs above"; fi; \
	   ${RM} -f ex5_1.tmp
#
#   The arco sparse matrices, as well as a few others are available at
# ftp://info.mcs.anl.gov/pub/petsc/testmatrices
#
runex6:
	-@${MPIRUN} -np 1 ex6 -ksp_type preonly  -pc_type lu -optionsleft_off \
	   -f ${PETSC_DIR}/src/mat/examples/matrices/arco1 > ex6_1.tmp 2>&1; \
	   if (diff output/ex6_1.out ex6_1.tmp) then true; \
	   else echo "Possible problem with ex6_1, diffs above"; fi; \
	   ${RM} -f ex6_1.tmp
runex6_2:
	-@${MPIRUN} -np 1 ex6 -sub_pc_type ilu -optionsleft_off \
	   -f ${PETSC_DIR}/src/mat/examples/matrices/arco1 \
	   -ksp_gmres_restart 100 -sub_ksp_type preonly -pc_type bjacobi \
	   -pc_bjacobi_blocks 8 -sub_pc_ilu_in_place -ksp_smonitor \
	   > ex6_2.tmp 2>&1; \
	   if (diff output/ex6_2.out ex6_2.tmp) then true; \
	   else echo "Possible problem with ex6_2, diffs above"; fi; \
	   ${RM} -f ex6_2.tmp
runex6_3:
	-@${MPIRUN} -np 1 ex6 -pc_type ilu -optionsleft_off \
	   -f ${PETSC_DIR}/src/mat/examples/matrices/arco4 \
	   -ksp_gmres_restart 100  -ksp_smonitor  > ex6_3.tmp 2>&1; \
	   if (diff output/ex6_3.out ex6_3.tmp) then true; \
	   else echo "Possible problem with ex6_3, diffs above"; fi; \
	   ${RM} -f ex6_3.tmp

# Note: We test these BlockSolve95 examples differently from the usual
# because the coloring heuristic is machine-dependent; convergence rates
# may differ slightly on various architectures.
runex6_6:
	-@${MPIRUN} -np 3 ex6  -ksp_gmres_restart 100 \
	   -f ${PETSC_DIR}/src/mat/examples/matrices/arco1_ns \
	   -mat_mpirowbs -pc_type ilu -pc_ilu_factorpointwise 
	-@echo "ex6_6: This program tested the ILU preconditioner in BlockSolve95."
runex6_7:
	-@${MPIRUN} -np 1 ex6 -pc_type asm -pc_asm_blocks 6 \
	   -f ${PETSC_DIR}/src/mat/examples/matrices/small \
	   -matload_block_size 6  -ksp_smonitor  > ex6_7.tmp 2>&1; \
	   if (diff output/ex6_7.out ex6_7.tmp) then true; \
	   else echo "Possible problem with ex6_7, diffs above"; fi; \
	   ${RM} -f ex6_7.tmp
runex7:
	-@${MPIRUN} -np 2 ex7 -pc_type jacobi -ksp_smonitor > ex7_1.tmp 2>&1; \
	   if (diff output/ex7_1.out ex7_1.tmp) then true; \
	   else echo "Possible problem with ex7_1, diffs above"; fi; \
	   ${RM} -f ex7_1.tmp
runex8:
	-@${MPIRUN} -np 1 ex8 -pc_type jacobi -ksp_smonitor > ex8_1.tmp 2>&1; \
	   if (diff output/ex8_1.out ex8_1.tmp) then true; \
	   else echo "Possible problem with ex8_1, diffs above"; fi; \
	   ${RM} -f ex8_1.tmp
runex8_2:
	-@${MPIRUN} -np 2 ex8 -pc_type jacobi -ksp_smonitor \
	   -ksp_rtol .000001 > ex8_2.tmp 2>&1;   \
	   if (diff output/ex8_2.out ex8_2.tmp) then true; \
	   else echo "Possible problem with ex8_2, diffs above"; fi; \
	   ${RM} -f ex8_2.tmp

# Note: We test these BlockSolve95 examples differently from the usual
# because the coloring heuristic is machine-dependent; convergence rates
# may differ slightly on various architectures.
runex8_3:
	-@${MPIRUN} -np 2 ex8 -pc_type icc -mat_mpirowbs -ksp_rtol .000001;\
	echo "ex8_3: This program tested the ICC preconditioner in BlockSolve95."
runex8_4:
	-@${MPIRUN} -np 2 ex8 -pc_type ilu -mat_nonsym -mat_mpirowbs \
	   -ksp_rtol .000001;\
	echo "ex8_4: This program tested the ILU preconditioner in BlockSolve95."
runex10:
	-@${MPIRUN} -np 1 ex10 -mat_bdiag -ksp_smonitor > ex10_1.tmp 2>&1; \
	   if (diff output/ex10_1.out ex10_1.tmp) then true; \
	   else echo "Possible problem with ex10_1, diffs above"; fi; \
	   ${RM} -f ex10_1.tmp
runex11:
	-@${MPIRUN} -np 3  ex11 -ksp_smonitor > ex11_1.tmp 2>&1; \
	   if (diff output/ex11_1.out ex11_1.tmp) then true; \
	   else echo "Possible problem with ex11_1, diffs above"; fi; \
	   ${RM} -f ex11_1.tmp
runex12:
	-@${MPIRUN} -np 3  ex12 -f ${PETSC_DIR}/src/mat/examples/matbinary.ex
runex13:
	-@${MPIRUN} -np 2  ex13 -ksp_smonitor > ex13_1.tmp 2>&1; \
	   if (diff output/ex13_1.out ex13_1.tmp) then true; \
	   else echo "Possible problem with ex13_1, diffs above"; fi; \
	   ${RM} -f ex13_1.tmp
runex17:
	-@${MPIRUN} -np 1 ex17 -n 6 -ksp_smonitor -p 5 -norandom -ksp_type gmres \
	   -pc_type jacobi > ex17_1.tmp 2>&1; \
	   if (diff output/ex17_1.out ex17_1.tmp) then true; \
	   else echo "Possible problem with ex17_1, diffs above"; fi; \
	   ${RM} -f ex17_1.tmp
runex18:
	-@${MPIRUN} -np 1 ex18 -f  ${PETSC_DIR}/src/mat/examples/matbinary.ex
runex19:
	-@${MPIRUN} -np 1 ex19 -t 2 -pc_type jacobi -ksp_smonitor -ksp_type gmres \
	   -s2_ksp_type bcgs -s2_pc_type jacobi -s2_ksp_smonitor \
           > ex19_1.tmp 2>&1; \
	   if (diff output/ex19_1.out ex19_1.tmp) then true; \
	   else echo "Possible problem with ex19_1, diffs above"; fi; \
	   ${RM} -f ex19_1.tmp
runex20:
	-@${MPIRUN} -np 1 ex20 -pc_type jacobi -mat_view -ksp_smonitor > ex20_1.tmp 2>&1; \
	   if (diff output/ex20_1.out ex20_1.tmp) then true; \
	   else echo "Possible problem with ex20_1, diffs above"; fi; \
	   ${RM} -f ex20_1.tmp

runex22:
	-@${MPIRUN} -np 1 ex22 -mat_aij -mat_view_info_detailed -ksp_smonitor > ex22_1.tmp 2>&1; \
	   if (diff output/ex22_1.out ex22_1.tmp) then true; \
	   else echo "Possible problem with ex22_1, diffs above"; fi; \
	   ${RM} -f ex22_1.tmp

runex22_2:
	-@${MPIRUN} -np 4 ex22 -m_eta 10 -m_xi 11 -mat_aij -mat_view_info \
	   -ksp_smonitor -pc_type bjacobi > ex22_2.tmp 2>&1;	  \
	   if (diff output/ex22_2.out ex22_2.tmp) then true; \
	   else echo "Possible problem with ex22_2, diffs above"; fi; \
	   ${RM} -f ex22_2.tmp

