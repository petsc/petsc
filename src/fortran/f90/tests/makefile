
ALL: lib

CFLAGS     =
FFLAGS     =
EXAMPLESC  = ex1.c test_IRIX.c
EXAMPLESF  = ex1f90.F test_IRIXf90.F
OBJSF      = 
LIBBASE    = libpetscfortran
LOCDIR     = src/fortran/f90/tests/
TESTS      = test_IRIX
CLEANFILES =
MANSEC     = Sys

include ${PETSC_DIR}/bmake/common/base

ex1f90: ex1.o ex1f90.o chkopts
	-${FLINKER} -o ex1f90 ex1f90.o ex1.o ${PETSC_SYS_LIB}
	${RM} -f ex1f90.o ex1.o

test_IRIX: test_IRIX.o test_IRIXf90.o
	-${FLINKER} -o test_IRIX test_IRIX.o test_IRIXf90.o
	${RM} -f test_IRIX.o test_IRIXf90.o

#
#-----------------------------------------------------------------------
#

runex1f90:
	-@./ex1f90 > ex1f90_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex1f90_1.out ex1f90_1.tmp) then true; \
	   else echo "Possible problem with ex1f90_1, diffs above"; fi; \
	   ${RM} -f ex1f90_1.tmp

TESTEXAMPLES_12 = ex1f90.PETSc runex1f90 ex1f90.rm

include ${PETSC_DIR}/bmake/common/test
