
ALL: 

CFLAGS	      = -fopenmp
FFLAGS	      =
CPPFLAGS      =
FPPFLAGS      =
LOCDIR        = src/benchmarks/streams/
EXAMPLESC     = BasicVersion.c OpenMPVersion.c SSEVersion.c PthreadVersion.c
EXAMPLESF     = 
TESTS         = BasicVersion OpenMPVersion
MANSEC        = Sys

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

BasicVersion: BasicVersion.o  chkopts 
	-@${CLINKER} -o BasicVersion BasicVersion.o ${PETSC_LIB}
	@${RM} -f BasicVersion.o

OpenMPVersion: OpenMPVersion.o  chkopts 
	-@${CLINKER} -o OpenMPVersion OpenMPVersion.o ${PETSC_LIB}
	@${RM} -f OpenMPVersion.o

SSEVersion: SSEVersion.o chkopts
	-${CLINKER} -o $@ $< ${PETSC_LIB}
	${RM} -f $<
PthreadVersion: PthreadVersion.o  chkopts 
	-@${CLINKER} -o PthreadVersion PthreadVersion.o ${PETSC_LIB}
	@${RM} -f PthreadVersion.o

test:  BasicVersion OpenMPVersion
	@if [ "${NP}foo" = "foo" ]; then echo "---------"; echo " Run with make streams NP=<integer number of cores to use>"; exit 1 ; fi
	-@echo "------------------------------------------------"
	-@${MPIEXEC} -n 1 ./BasicVersion
	-@echo "------------------------------------------------"
	-@${MPIEXEC} -n ${NP} ./BasicVersion
	-@echo "------------------------------------------------"
	-@OMP_NUM_THREADS=${NP}; export OMP_NUM_THREADS; ${MPIEXEC} -n 1 ./OpenMPVersion
	-@echo "------------------------------------------------"
