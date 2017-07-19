CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/label/examples/tutorials/
MANSEC          = DM
SUBMANSEC       = DMLABEL

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o   chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_DM_LIB}
	${RM} -f ex1.o

ex1f90: ex1f90.o  chkopts
	-${FLINKER} -o ex1f90 ex1f90.o ${PETSC_DM_LIB}
	${RM} -f ex1f90.o
#-----------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 2 ./ex1 -nox

runex1f90:
	-@${MPIEXEC} -n 2 ./ex1f90 -nox

include ${PETSC_DIR}/lib/petsc/conf/test
