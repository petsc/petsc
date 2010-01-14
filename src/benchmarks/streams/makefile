
ALL: 

CFLAGS	      =
FFLAGS	      =	 
CPPFLAGS      =
FPPFLAGS      =
LOCDIR        = src/benchmarks/streams/
EXAMPLESC     = BasicVersion.c OpenMPVersion.c
EXAMPLESF     = 
TESTS         = BasicVersion OpenMPVersion
MANSEC        = Sys

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

BasicVersion: BasicVersion.o  chkopts 
	-${CLINKER} -o BasicVersion BasicVersion.o ${PETSC_LIB}
	${RM} -f BasicVersion.o

OpenMPVersion: OpenMPVersion.o  chkopts 
	-${CLINKER} -o OpenMPVersion OpenMPVersion.o ${PETSC_LIB}
	${RM} -f OpenMPVersion.o

test:  BasicVersion OpenMPVersion
	-@echo "Time Taken by STREAMS as follows"
	-@echo "------------------------------------------------"
	-@${MPIEXEC} -n 1 ./BasicVersion
	-@read -p "Enter number of OpenMP threads to use: " cnt ; OMP_NUM_THREADS=${cnt}; export OMP_NUM_THREADS; ${MPIEXEC} -n 1 ./OpenMPVersion
	-@echo "------------------------------------------------"
