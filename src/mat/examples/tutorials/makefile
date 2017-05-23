
CFLAGS 	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/mat/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c ex5.c ex8.c ex9.c ex10.c ex11.c ex12.c ex15.c ex16.c ex17.c
EXAMPLESF       =
MANSEC          = Mat

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER}  -o ex1 ex1.o ${PETSC_MAT_LIB}
	${RM} ex1.o

ex2: ex2.o  chkopts
	-${CLINKER}  -o ex2 ex2.o ${PETSC_MAT_LIB}
	${RM} ex2.o

ex3: ex3.o  chkopts
	-${CLINKER}  -o ex3 ex3.o ${PETSC_MAT_LIB}
	${RM} ex3.o

ex2_kds: ex2_kds.o  chkopts
	-${CLINKER}  -o ex2_kds ex2_kds.o ${PETSC_MAT_LIB}
	${RM} ex2_kds.o

ex5: ex5.o  chkopts
	-${CLINKER}  -o ex5 ex5.o ${PETSC_MAT_LIB}
	${RM} ex5.o

ex8: ex8.o  chkopts
	-${CLINKER}  -o ex8 ex8.o ${PETSC_MAT_LIB}
	${RM} ex8.o
ex9: ex9.o  chkopts
	-${CLINKER}  -o ex9 ex9.o ${PETSC_MAT_LIB}
	${RM} ex9.o
ex10: ex10.o  chkopts
	-${CLINKER}  -o ex10 ex10.o ${PETSC_MAT_LIB}
	${RM} ex10.o
ex11: ex11.o  chkopts
	-${CLINKER}  -o ex11 ex11.o ${PETSC_MAT_LIB}
	${RM} ex11.o
ex12: ex12.o  chkopts
	-${CLINKER}  -o ex12 ex12.o ${PETSC_MAT_LIB}
	${RM} ex12.o
ex13: ex13.o  chkopts
	-${CLINKER}  -o ex13 ex13.o ${PETSC_MAT_LIB}
	${RM} ex13.o

ex14: ex14.o  chkopts
	-${CLINKER}  -o ex14 ex14.o ${PETSC_DM_LIB}
	${RM} ex14.o

ex15: ex15.o  chkopts
	-${CLINKER}  -o ex15 ex15.o ${PETSC_MAT_LIB}
	${RM} ex15.o

ex16: ex16.o  chkopts
	-${CLINKER}  -o ex16 ex16.o ${PETSC_MAT_LIB}
	${RM} ex16.o

ex17: ex17.o  chkopts
	-${CLINKER}  -o ex17 ex17.o ${PETSC_MAT_LIB}
	${RM} ex17.o

# See http://www.mcs.anl.gov/petsc/documentation/faq.html#datafiles for how to obtain the datafiles used below
runex1:
	-@${MPIEXEC} -n 1 ./ex1 -f0 ${DATAFILESPATH}/matrices/medium -viewer_binary_skip_info

runex5:
	-@${MPIEXEC} -n 2 ./ex5 -f ${DATAFILESPATH}/matrices/medium -viewer_binary_skip_info

runex8:
	-@${MPIEXEC} -n 1 ./ex8 > ex8_1.tmp 2>&1;   \
	   if (${DIFF} output/ex8_1.out ex8_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex8, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex8_1.tmp

runex8_2:
	-@${MPIEXEC} -n 2 ./ex8 > ex8_2.tmp 2>&1;   \
	   if (${DIFF} output/ex8_2.out ex8_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex8_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex8_2.tmp

runex9:
	-@${MPIEXEC} -n 2 ./ex9 -f ${DATAFILESPATH}/matrices/medium -viewer_binary_skip_info

runex10_seqaij:
	-@${MPIEXEC} -n 1 ./ex10 -f ${DATAFILESPATH}/matrices/small -a_mat_type seqaij  > ex10_seqaij.tmp 2>&1;   \
	   if (${DIFF} output/ex10_seqaij.out ex10_seqaij.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex10_seqaij, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex10_seqaij.tmp

runex10_seqdense:
	-@${MPIEXEC} -n 1 ./ex10 -f ${DATAFILESPATH}/matrices/small -a_mat_type seqdense > ex10_seqdense.tmp 2>&1;   \
	   if (${DIFF} output/ex10_seqdense.out ex10_seqdense.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex10_seqdense, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex10_seqdense.tmp

runex10_mpiaij:
	-@${MPIEXEC} -n 2 ./ex10 -f ${DATAFILESPATH}/matrices/small -a_mat_type mpiaij > ex10_mpiaij.tmp 2>&1;   \
	   if (${DIFF} output/ex10_mpiaij.out ex10_mpiaij.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex10_mpiaij, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex10_mpiaij.tmp

runex10_mpidense:
	-@${MPIEXEC} -n 2 ./ex10 -f ${DATAFILESPATH}/matrices/small -a_mat_type mpidense > ex10_mpidense.tmp 2>&1;   \
	   if (${DIFF} output/ex10_mpidense.out ex10_mpidense.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex10_mpidense, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex10_mpidense.tmp

runex11:
	-@${MPIEXEC} -n 2 ./ex11 > ex11_1.tmp 2>&1;   \
	   if (${DIFF} output/ex11_1.out ex11_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex11, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex11_1.tmp

runex12:
	-@${MPIEXEC} -n 1 ./ex12 -f0 ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 > ex12_1.tmp 2>&1;   \
	   if (${DIFF} output/ex12.out ex12_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex12, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex12_1.tmp

runex15:
	-@x="bad"; ${MPIEXEC} -n 3 ./ex15 -mat_partitioning_type parmetis > ex15_1.tmp 2>&1;   \
	   if (${DIFF} output/ex15_1.out ex15_1.tmp > /dev/null 2>&1) then x='good'; fi; \
	   if (${DIFF} output/ex15_1_alt.out ex15_1.tmp > /dev/null 2>&1) then x='good'; fi; \
           if [ "$$x" = "bad" ]; then ${DIFF} output/ex15_1.out ex15_1.tmp ; ${DIFF} output/ex15_1_alt.out ex15_1.tmp ; printf "${PWD}\nPossible problem with ex15, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex15_1.tmp

runex15_2:
	-@${MPIEXEC} -n 3 ./ex15 -mat_partitioning_type ptscotch > ex15_2.tmp 2>&1;   \
	   if (${DIFF} output/ex15_2.out ex15_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex15_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex15_2.tmp

runex17:
	-@${MPIEXEC} -n 8 ./ex17 -emptyranks 0,2,4 -bigranks 1,3,7 > ex17_1.tmp 2>&1;   \
	   ${DIFF} output/ex17_1.out ex17_1.tmp || printf "${PWD}\nPossible problem with ex17_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex17_1.tmp

#-------------------------------------------------------------
TESTEXAMPLES_C		    = ex5.PETSc ex5.rm ex10.PETSc ex8.PETSc runex8 runex8_2 ex8.rm  ex10.rm ex12.PETSc ex12.rm
TESTEXAMPLES_C_X	    = ex2.PETSc ex2.rm
TESTEXAMPLES_DOUBLEINT32    = ex12.PETSc runex12 ex12.rm
TESTEXAMPLES_FORTRAN	    =
TESTEXAMPLES_C_X_MPIUNI     =
TESTEXAMPLES_DATAFILESPATH  = ex1.PETSc runex1 ex1.rm ex5.PETSc runex5 ex5.rm ex9.PETSc runex9 ex9.rm ex10.PETSc runex10_seqaij runex10_seqdense runex10_mpiaij runex10_mpidense ex10.rm
TESTEXAMPLES_FORTRAN_MPIUNI =
TESTEXAMPLES_13		    = ex1.PETSc ex1.rm
TESTEXAMPLES_PARMETIS      = ex11.PETSc runex11 ex11.rm ex15.PETSc runex15 ex15.rm
TESTEXAMPLES_PTSCOTCH       = ex15.PETSc runex15_2 ex15.rm


include ${PETSC_DIR}/lib/petsc/conf/test

