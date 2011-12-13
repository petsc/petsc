
CFLAGS	         =
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR           = src/vec/bg/examples/tutorials/
EXAMPLESC        = ex1.c
EXAMPLESF        =

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_VEC_LIB}
	${RM} -f ex1.o

#------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1  ./ex1  > ex1.tmp 2>&1; \
	   ${DIFF} output/ex1.out ex1.tmp || echo "${PWD}\n Possible problem with with ex1, diffs above \n========================================="; \
	   ${RM} -f ex1.tmp

TESTEXAMPLES_C		    = ex1.PETSc runex1 ex1.rm
TESTEXAMPLES_C_X11	    =
TESTEXAMPLES_FORTRAN	    =
TESTEXAMPLES_FORTRAN_MPIUNI =
TESTEXAMPLES_C_X11_MPIUNI   =
TESTEXAMPLES_F90	    =

include ${PETSC_DIR}/conf/test
