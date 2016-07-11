
CFLAGS	         =
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR   = src/vec/is/sf/examples/tutorials/
EXAMPLESC        = ex1.c ex2.c
EXAMPLESF        =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_VEC_LIB}
	${RM} -f ex1.o

ex2: ex2.o chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_VEC_LIB}
	${RM} -f ex2.o

ex3: ex3.o chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_VEC_LIB}
	${RM} -f ex3.o

#------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 4 ./ex1 -test_bcast -sf_type window > ex1_1.tmp 2>&1; \
	   ${DIFF} output/ex1_1.out ex1_1.tmp || printf "${PWD}\nPossible problem with ex1, diffs above\n=========================================\n"; \
	   ${RM} -f ex1_1.tmp
runex1_basic:
	-@${MPIEXEC} -n 4 ./ex1 -test_bcast -sf_type basic  > ex1_1.tmp 2>&1; \
	   ${DIFF} output/ex1_1_basic.out ex1_1.tmp || printf "${PWD}\nPossible problem with ex1_basic, diffs above\n=========================================\n"; \
	   ${RM} -f ex1_1.tmp
runex1_2:
	-@${MPIEXEC} -n 4 ./ex1 -test_reduce -sf_type window> ex1_2.tmp 2>&1; \
	   ${DIFF} output/ex1_2.out ex1_2.tmp || printf "${PWD}\nPossible problem with ex1_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex1_2.tmp
runex1_2_basic:
	-@${MPIEXEC} -n 4 ./ex1 -test_reduce -sf_type basic > ex1_2.tmp 2>&1; \
	   ${DIFF} output/ex1_2_basic.out ex1_2.tmp || printf "${PWD}\nPossible problem with ex1_2_basic, diffs above\n=========================================\n"; \
	   ${RM} -f ex1_2.tmp
runex1_3:
	-@${MPIEXEC} -n 4 ./ex1 -test_degree -sf_type window > ex1_3.tmp 2>&1; \
	   ${DIFF} output/ex1_3.out ex1_3.tmp || printf "${PWD}\nPossible problem with ex1_3, diffs above\n=========================================\n"; \
	   ${RM} -f ex1_3.tmp
runex1_3_basic:
	-@${MPIEXEC} -n 4 ./ex1 -test_degree -sf_type basic  > ex1_3.tmp 2>&1; \
	   ${DIFF} output/ex1_3_basic.out ex1_3.tmp || printf "${PWD}\nPossible problem with ex1_3_basic, diffs above\n=========================================\n"; \
	   ${RM} -f ex1_3.tmp
runex1_4:
	-@${MPIEXEC} -n 4 ./ex1 -test_gather -sf_type window > ex1_4.tmp 2>&1; \
	   ${DIFF} output/ex1_4.out ex1_4.tmp || printf "${PWD}\nPossible problem with ex1_4, diffs above\n=========================================\n"; \
	   ${RM} -f ex1_4.tmp
runex1_4_basic:
	-@${MPIEXEC} -n 4 ./ex1 -test_gather -sf_type basic  > ex1_4.tmp 2>&1; \
	   ${DIFF} output/ex1_4_basic.out ex1_4.tmp || printf "${PWD}\nPossible problem with ex1_4_basic, diffs above\n=========================================\n"; \
	   ${RM} -f ex1_4.tmp
runex1_4_stride:
	-@${MPIEXEC} -n 4 ./ex1 -test_gather -sf_type basic -stride 2 > ex1_4.tmp 2>&1; \
	   ${DIFF} output/ex1_4_stride.out ex1_4.tmp || printf "${PWD}\nPossible problem with ex1_4_stride, diffs above\n=========================================\n"; \
	   ${RM} -f ex1_4.tmp
runex1_5:
	-@${MPIEXEC} -n 4 ./ex1 -test_scatter -sf_type window > ex1_5.tmp 2>&1; \
	   ${DIFF} output/ex1_5.out ex1_5.tmp || printf "${PWD}\nPossible problem with ex1_5, diffs above\n=========================================\n"; \
	   ${RM} -f ex1_5.tmp
runex1_5_basic:
	-@${MPIEXEC} -n 4 ./ex1 -test_scatter -sf_type basic  > ex1_5.tmp 2>&1; \
	   ${DIFF} output/ex1_5_basic.out ex1_5.tmp || printf "${PWD}\nPossible problem with ex1_5_basic, diffs above\n=========================================\n"; \
	   ${RM} -f ex1_5.tmp
runex1_5_stride:
	-@${MPIEXEC} -n 4 ./ex1 -test_scatter -sf_type basic -stride 2 > ex1_5.tmp 2>&1; \
	   ${DIFF} output/ex1_5_stride.out ex1_5.tmp || printf "${PWD}\nPossible problem with ex1_5_stride, diffs above\n=========================================\n"; \
	   ${RM} -f ex1_5.tmp
runex1_6:
	-@${MPIEXEC} -n 4 ./ex1 -test_embed -sf_type window > ex1_6.tmp 2>&1; \
	   ${DIFF} output/ex1_6.out ex1_6.tmp || printf "${PWD}\nPossible problem with ex1_6, diffs above\n=========================================\n"; \
	   ${RM} -f ex1_6.tmp
runex1_6_basic:
	-@${MPIEXEC} -n 4 ./ex1 -test_embed -sf_type basic > ex1_6.tmp 2>&1; \
	   ${DIFF} output/ex1_6_basic.out ex1_6.tmp || printf "${PWD}\nPossible problem with ex1_6_basic, diffs above\n=========================================\n"; \
	   ${RM} -f ex1_6.tmp
runex1_7:
	-@${MPIEXEC} -n 4 ./ex1 -test_invert -sf_type window > ex1_7.tmp 2>&1; \
	   ${DIFF} output/ex1_7.out ex1_7.tmp || printf "${PWD}\nPossible problem with ex1_7, diffs above\n=========================================\n"; \
	   ${RM} -f ex1_7.tmp
runex1_7_basic:
	-@${MPIEXEC} -n 4 ./ex1 -test_invert -sf_type basic > ex1_7.tmp 2>&1; \
	   ${DIFF} output/ex1_7_basic.out ex1_7.tmp || printf "${PWD}\nPossible problem with ex1_7_basic, diffs above\n=========================================\n"; \
	   ${RM} -f ex1_7.tmp

runex2_basic:
	-@${MPIEXEC} -n 2 ./ex2 -sf_type basic > ex2.tmp 2>&1; \
          ${DIFF} output/ex2_basic.out ex2.tmp || printf "${PWD}\nPossible problem with ex2_basic, diffs above\n=========================================\n"; \
          ${RM} -f ex2.tmp

runex2_window:
	-@${MPIEXEC} -n 2 ./ex2 -sf_type window > ex2.tmp 2>&1; \
          ${DIFF} output/ex2_window.out ex2.tmp || printf "${PWD}\nPossible problem with ex2_window, diffs above\n=========================================\n"; \
          ${RM} -f ex2.tmp

runex3_basic:
	-@${MPIEXEC} -n 1 ./ex3 -sf_type basic > ex3.tmp 2>&1; \
	  ${DIFF} output/ex3_basic.out ex3.tmp || printf "${PWD}\nPossible problem with ex3_basic, diffs above\n=========================================\n"; \
	  ${RM} -f ex3.tmp

runex3_window:
	-@${MPIEXEC} -n 1 ./ex3 -sf_type window > ex3.tmp 2>&1; \
	  ${DIFF} output/ex3_window.out ex3.tmp || printf "${PWD}\nPossible problem with ex3_window, diffs above\n=========================================\n"; \
	  ${RM} -f ex3.tmp

runex3_basic_dupped:
	-@${MPIEXEC} -n 1 ./ex3 -test_dupped_type -sf_type basic > ex3.tmp 2>&1; \
	  ${DIFF} output/ex3_basic_dupped.out ex3.tmp || printf "${PWD}\nPossible problem with ex3_basic_dupped, diffs above\n=========================================\n"; \
	  ${RM} -f ex3.tmp

runex3_window_dupped:
	-@${MPIEXEC} -n 1 ./ex3 -test_dupped_type -sf_type window > ex3.tmp 2>&1; \
	  ${DIFF} output/ex3_window_dupped.out ex3.tmp || printf "${PWD}\nPossible problem with ex3_window_dupped, diffs above\n=========================================\n"; \
	  ${RM} -f ex3.tmp

TESTEXAMPLES_C		    = ex1.PETSc runex1_basic runex1_2_basic runex1_3_basic runex1_4_basic runex1_4_stride runex1_5_basic runex1_5_stride runex1_6_basic runex1_7_basic ex1.rm \
                                ex2.PETSc runex2_basic runex2_window ex2.rm ex3.PETSc runex3_basic runex3_window runex3_basic_dupped runex3_window_dupped ex3.rm
TESTEXAMPLES_C_X	    =
TESTEXAMPLES_FORTRAN	    =
TESTEXAMPLES_FORTRAN_MPIUNI =
TESTEXAMPLES_C_X_MPIUNI   =
TESTEXAMPLES_F90	    =

include ${PETSC_DIR}/lib/petsc/conf/test
