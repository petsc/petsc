
CFLAGS	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sys/src/viewer/examples/tests/
EXAMPLESC       = ex3.c
MANSEC          = Viewer

include ${PETSC_DIR}/bmake/common/base

ex3: ex3.o   chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_SYS_LIB}
	${RM} -f ex3.o

runex3:
	-@${MPIRUN} -np 1 ./ex3

TESTEXAMPLES_2  = 
TESTEXAMPLES_4  = ex3.PETSc runex3 ex3.rm

include ${PETSC_DIR}/bmake/common/test

