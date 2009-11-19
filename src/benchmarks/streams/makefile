
ALL: 

CFLAGS	      =
FFLAGS	      =	 
CPPFLAGS      =
FPPFLAGS      =
LOCDIR        = src/benchmarks/streams
EXAMPLESC     = BasicVersion.c OpenMPVersion.c
EXAMPLESF     = 
TESTS         = BasicVersion OpenMPVersion
MANSEC        = Sys

include ${PETSC_DIR}/conf/base
include ${PETSC_DIR}/conf/test

BasicVersion: BasicVersion.o  chkopts 
	-${CLINKER} -o BasicVersion BasicVersion.o ${PETSC_LIB}
	${RM} -f BasicVersion.o

OpenMPVersion: OpenMPVersion.o  chkopts 
	-${CLINKER} -o OpenMPVersion OpenMPVersion.o ${PETSC_LIB}
	${RM} -f OpenMPVersion.o

test: ${TESTS}

runtest: 
	-@echo "Time Taken by STREAMS as follows"
	-@echo "------------------------------------------------"
	-@${MPIEXEC} -n 1 ./BasicVersion
	-@${MPIEXEC} -n 1 ./OpenMPVersion
	-@echo "------------------------------------------------"
