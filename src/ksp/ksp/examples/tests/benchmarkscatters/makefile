
CFLAGS	  =
FFLAGS    =
CPPFLAGS  =
FPPFLAGS  =
LOCDIR    = src/ksp/ksp/examples/tests/benchmarkscatters/
EXAMPLESC       = ex1.c ex2.c

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_KSP_LIB}
	${RM} ex1.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_KSP_LIB}
	${RM} ex2.o

include ${PETSC_DIR}/lib/petsc/conf/test
