-include ../../../petscdir.mk

LOCDIR        = src/benchmarks/streams/
MANSEC        = Sys

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

BasicVersion: BasicVersion.o
	-@${CLINKER} -o BasicVersion BasicVersion.o ${PETSC_LIB}
	@${RM} -f BasicVersion.o

MPIVersion: MPIVersion.o
	-@${CLINKER} -o MPIVersion MPIVersion.o ${PETSC_LIB}
	@${RM} -f MPIVersion.o

CUDAVersion: CUDAVersion.o
	-@${CLINKER} -o CUDAVersion CUDAVersion.o ${PETSC_LIB}
	@${RM} -f CUDAVersion.o

OpenMPVersion: OpenMPVersion.o
	-@${CLINKER} -o OpenMPVersion OpenMPVersion.o
	@${RM} -f OpenMPVersion.o

SSEVersion: SSEVersion.o
	-${CLINKER} -o $@ $< ${PETSC_LIB}
	${RM} -f $<

PthreadVersion: PthreadVersion.o
	-@${CLINKER} -o PthreadVersion PthreadVersion.o ${PETSC_LIB}
	@${RM} -f PthreadVersion.o

# make streams [NPMAX=integer_number_of_MPI_processes_to_use] [MPI_BINDING='binding options']
mpistream:  MPIVersion
	@if [ "${NPMAX}foo" = "foo" ]; then echo "---------"; printf " Run with [PETSC_OPTIONS=-process_view] make streams NPMAX=<integer number of MPI processes> [MPI_BINDING='-bind-to core -map-by numa']\n or       [I_MPI_PIN_PROCESSOR_LIST=:map=scatter] [PETSC_OPTIONS=-process_view] make streams NPMAX=<integer number of MPI processes>\n"; exit 1 ; fi
	-@printf "" > scaling.log
	-@printf "Running streams with '${MPIEXEC} ${MPI_BINDING}' using 'NPMAX=${NPMAX}' \n"
	-@i=0; while [ $${i} -lt ${NPMAX} ]; do i=`expr $${i} + 1`; \
	  ${MPIEXEC} ${MPI_BINDING} -n $${i} ./MPIVersion | tee -a scaling.log; \
        done
	-@echo "------------------------------------------------"
	-@${PYTHON} process.py MPI fileoutput

# Works on SUMMIT
cudastreamjsrun:  CUDAVersion
	@if [ "${NPMAX}foo" = "foo" ]; then echo "---------"; printf " Run with [PETSC_OPTIONS=-process_view] make streams NPMAX=<integer number of MPI processes> [MPI_BINDING='-bind-to core -map-by numa']\n or       [I_MPI_PIN_PROCESSOR_LIST=:map=scatter] [PETSC_OPTIONS=-process_view] make streams NPMAX=<integer number of MPI processes>\n"; exit 1 ; fi
	-@printf "" > scaling.log
	-@printf "Running streams with '${MPIEXEC} ${MPI_BINDING}' using 'NPMAX=${NPMAX}' \n"
	-@i=0; while [ $${i} -lt ${NPMAX} ] && [ $${i} -lt 7 ]; do i=`expr $${i} + 1`; \
	  ${MPIEXEC} ${MPI_BINDING} -n 1 -c$${i} -a$${i} -g1 ./CUDAVersion | tee -a scaling.log; \
        done
	-@n=1; i=7; while [ $${i} -lt ${NPMAX} ]; do i=`expr $${i} + 7`; n=`expr $${n} + 1`; \
	       c=5; while [ $${c} -lt 7 ]; do c=`expr $${c} + 1`; \
	  ${MPIEXEC} ${MPI_BINDING} -n $${n} -c$${c} -a$${c} -g1 ./CUDAVersion | tee -a scaling.log; \
        done; done
	-@echo "------------------------------------------------"
	-@${PYTHON} process.py CUDA fileoutput

openmpstream:  OpenMPVersion
	@if [ "${NPMAX}foo" = "foo" ]; then echo "---------"; printf " Run with make openmpstream NPMAX=<integer number of threads>\n"; exit 1 ; fi
	-@printf "" > scaling.log
	@-@printf "Running openmpstreams using 'NPMAX=${NPMAX}'\n"
	-@i=0; while [ $${i} -lt ${NPMAX} ]; do i=`expr $${i} + 1`; \
	  OMP_NUM_THREADS=$${i} ./OpenMPVersion  | tee -a scaling.log;\
        done
	-@${PYTHON} process.py OpenMP fileoutput

hwloc:
	-@if [ "${LSTOPO}foo" != "foo" ]; then ${MPIEXEC} ${MPI_BINDING} -n 1 ${LSTOPO} --no-icaches --no-io --ignore PU ; fi

mpistreams: mpistream hwloc
	-@${PYTHON} process.py MPI


openmpstreams: openmpstream hwloc
	-@${PYTHON} process.py OpenMP


