
ALL:

CFLAGS	      = 
FFLAGS	      =
CPPFLAGS      =
FPPFLAGS      =
LOCDIR        = src/benchmarks/streams/
EXAMPLESC     = BasicVersion.c MPIVersion.c OpenMPVersion.c SSEVersion.c PthreadVersion.c
EXAMPLESF     =
TESTS         = BasicVersion OpenMPVersion
MANSEC        = Sys

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

BasicVersion: BasicVersion.o  chkopts
	-@${CLINKER} -o BasicVersion BasicVersion.o ${PETSC_LIB}
	@${RM} -f BasicVersion.o

MPIVersion: MPIVersion.o  chkopts
	-@${CLINKER} -o MPIVersion MPIVersion.o ${PETSC_LIB}
	@${RM} -f MPIVersion.o

OpenMPVersion: OpenMPVersion.o  chkopts
	-@${CLINKER} -o OpenMPVersion OpenMPVersion.o ${PETSC_LIB}
	@${RM} -f OpenMPVersion.o

SSEVersion: SSEVersion.o chkopts
	-${CLINKER} -o $@ $< ${PETSC_LIB}
	${RM} -f $<

PthreadVersion: PthreadVersion.o  chkopts
	-@${CLINKER} -o PthreadVersion PthreadVersion.o ${PETSC_LIB}
	@${RM} -f PthreadVersion.o

# make streams [NPMAX=integer_number_of_MPI_processes_to_use] [MPI_BINDING='binding options']
stream:  MPIVersion
	-@printf "" > scaling.log
	-@echo "Running streams with '${MPIEXEC} ${MPI_BINDING}' using 'NPMAX=${NPMAX}' "
	-@i=0; while [ $${i} -lt ${NPMAX} ]; do i=`expr $${i} + 1`; \
	  ${MPIEXEC} ${MPI_BINDING} -n $${i} ./MPIVersion | tee -a scaling.log; \
        done
	-@echo "------------------------------------------------"
	-@${PYTHON} process.py createfile

hwloc:
	-@if [ "${LSTOPO}foo" != "foo" ]; then ${MPIEXEC} ${MPI_BINDING} -n 1 ${LSTOPO} --no-icaches --no-io --ignore PU ; fi

streams: stream hwloc
	-@${PYTHON} process.py


