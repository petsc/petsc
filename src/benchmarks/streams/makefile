-include ../../../petscdir.mk

MANSEC        = Sys
#CFLAGS        = -mcmodel=large
# The CFLAGS above allows very large global arrays

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

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

OpenMPVersionLikeMPI: OpenMPVersionLikeMPI.o
	-@${CLINKER} -o OpenMPVersionLikeMPI OpenMPVersionLikeMPI.o
	@${RM} -f OpenMPVersionLikeMPI.o

SSEVersion: SSEVersion.o
	-${CLINKER} -o $@ $< ${PETSC_LIB}
	${RM} -f $<

PthreadVersion: PthreadVersion.o
	-@${CLINKER} -o PthreadVersion PthreadVersion.o ${PETSC_LIB}
	@${RM} -f PthreadVersion.o

# If not set by users, use a binding that's good for both MPICH and Open MPI
MPI_BINDING ?= -map-by numa -bind-to core

# make streams [NPMAX=integer_number_of_MPI_processes_to_use] [MPI_BINDING='binding options']
mpistream:  MPIVersion
	@if [ "${NPMAX}foo" = "foo" ]; then echo "---------"; printf " Run with [PETSC_OPTIONS=-process_view] make streams NPMAX=<integer number of MPI processes> [MPI_BINDING='-bind-to core -map-by numa']\n or       [I_MPI_PIN_PROCESSOR_LIST=:map=scatter] [PETSC_OPTIONS=-process_view] make streams NPMAX=<integer number of MPI processes>\n"; exit 1 ; fi
	-@printf "" > scaling.log
	-@printf "Running streams with '${MPIEXEC} ${MPI_BINDING} ${MPI_BINDING_VIEW} -n <np> ./MPIVersion' using 'NPMAX=${NPMAX}' \n"
	-@printf "(Hint: To change MPI process binding, use env var MPI_BINDING. To visualize the binding, with Open MPI, use another env var, MPI_BINDING_VIEW=-display-map, see 'mpirun --help binding | mapping';"
	-@printf " with MPICH, use HYDRA_TOPO_DEBUG=1 instead, see 'mpirun -bind-to -help')\n"
	-@i=0; while [ $${i} -lt ${NPMAX} ]; do i=`expr $${i} + 1`; \
	  ${MPIEXEC} ${MPI_BINDING} ${MPI_BINDING_VIEW} -n $${i} ./MPIVersion | tee -a scaling.log; \
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
	-@printf "You can set OMP_DISPLAY_AFFINITY=true to show binding, and set OMP_PLACES, OMP_PROC_BIND properly to set binding \n"
	-@i=0; while [ $${i} -lt ${NPMAX} ]; do i=`expr $${i} + 1`; \
	  OMP_NUM_THREADS=$${i} ./OpenMPVersion  | tee -a scaling.log;\
        done
	-@${PYTHON} process.py OpenMP fileoutput

openmplikempistream:  OpenMPVersionLikeMPI
	@if [ "${NPMAX}foo" = "foo" ]; then echo "---------"; printf " Run with make openmplikempistream NPMAX=<integer number of threads>\n"; exit 1 ; fi
	-@printf "" > scaling.log
	@-@printf "Running openmplikempistreams using 'NPMAX=${NPMAX}'\n"
	-@i=0; while [ $${i} -lt ${NPMAX} ]; do i=`expr $${i} + 1`; \
	  OMP_NUM_THREADS=$${i} ./OpenMPVersionLikeMPI  | tee -a scaling.log;\
        done
	-@${PYTHON} process.py OpenMPLikeMPI fileoutput

hwloc:
	-@if [ "${LSTOPO}foo" != "foo" ]; then ${MPIEXEC} ${MPI_BINDING} -n 1 ${LSTOPO} --no-icaches --no-io --ignore PU ; fi

mpistreams: mpistream hwloc
	-@${PYTHON} process.py MPI


openmpstreams: openmpstream hwloc
	-@${PYTHON} process.py OpenMP
