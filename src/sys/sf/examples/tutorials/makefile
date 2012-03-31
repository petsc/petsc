
CFLAGS	         =
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR           = src/sys/sf/examples/tutorials/
EXAMPLESC        = ex1.c
EXAMPLESF        =

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_VEC_LIB}
	${RM} -f ex1.o

#------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 4 ./ex1 -test_bcast -sf_view_verbose > ex1_1.tmp 2>&1; \
	   ${DIFF} output/ex1_1.out ex1_1.tmp || echo "${PWD}\n Possible problem with with ex1, diffs above \n========================================="; \
	   ${RM} -f ex1_1.tmp
runex1_2:
	-@${MPIEXEC} -n 4 ./ex1 -test_reduce -sf_view_verbose > ex1_2.tmp 2>&1; \
	   ${DIFF} output/ex1_2.out ex1_2.tmp || echo "${PWD}\n Possible problem with with ex1_2, diffs above \n========================================="; \
	   ${RM} -f ex1_2.tmp
runex1_3:
	-@${MPIEXEC} -n 4 ./ex1 -test_degree -sf_view_verbose > ex1_3.tmp 2>&1; \
	   ${DIFF} output/ex1_3.out ex1_3.tmp || echo "${PWD}\n Possible problem with with ex1_3, diffs above \n========================================="; \
	   ${RM} -f ex1_3.tmp
runex1_4:
	-@${MPIEXEC} -n 4 ./ex1 -test_gather -sf_view_verbose > ex1_4.tmp 2>&1; \
	   ${DIFF} output/ex1_4.out ex1_4.tmp || echo "${PWD}\n Possible problem with with ex1_4, diffs above \n========================================="; \
	   ${RM} -f ex1_4.tmp
runex1_5:
	-@${MPIEXEC} -n 4 ./ex1 -test_scatter -sf_view_verbose > ex1_5.tmp 2>&1; \
	   ${DIFF} output/ex1_5.out ex1_5.tmp || echo "${PWD}\n Possible problem with with ex1_5, diffs above \n========================================="; \
	   ${RM} -f ex1_5.tmp
runex1_6:
	-@${MPIEXEC} -n 4 ./ex1 -test_embed -sf_view_verbose > ex1_6.tmp 2>&1; \
	   ${DIFF} output/ex1_6.out ex1_6.tmp || echo "${PWD}\n Possible problem with with ex1_6, diffs above \n========================================="; \
	   ${RM} -f ex1_6.tmp

TESTEXAMPLES_C		    = ex1.PETSc runex1 runex1_2 runex1_3 runex1_4 runex1_5 runex1_6 ex1.rm
TESTEXAMPLES_C_X	    =
TESTEXAMPLES_FORTRAN	    =
TESTEXAMPLES_FORTRAN_MPIUNI =
TESTEXAMPLES_C_X_MPIUNI   =
TESTEXAMPLES_F90	    =

include ${PETSC_DIR}/conf/test
