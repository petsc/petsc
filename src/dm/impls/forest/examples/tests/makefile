CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/impls/forest/examples/tests/
EXAMPLESC       = ex2.c
EXAMPLESF       =
MANSEC          = DM

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_DM_LIB}
	${RM} -f ex1.o

runex2_2d:
	-@${MPIEXEC} -n 1 ./ex2 -petscspace_poly_tensor -petscspace_order 1 -dim 2
runex2_3d:
	-@${MPIEXEC} -n 1 ./ex2 -petscspace_poly_tensor -petscspace_order 1 -dim 3

#--------------------------------------------------------------------------

include ${PETSC_DIR}/lib/petsc/conf/test
