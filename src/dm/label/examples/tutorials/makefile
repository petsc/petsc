CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/label/examples/tutorials/
MANSEC          = DM
SUBMANSEC       = DMLABEL

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
#-----------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 2 ./ex1 -nox

runex1f90:
	-@${MPIEXEC} -n 2 ./ex1f90 -nox

include ${PETSC_DIR}/lib/petsc/conf/test
