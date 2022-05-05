-include ../../petsckdir.mk

ALL:

LOCDIR    = src/benchmarks/
MANSEC    = Sys

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

PetscTime: PetscTime.o 
	-${CLINKER} -o PetscTime PetscTime.o ${PETSC_LIB}
	${RM} -f PetscTime.o

PetscGetTime: PetscGetTime.o 
	-${CLINKER} -o PetscGetTime PetscGetTime.o ${PETSC_LIB}
	${RM} -f PetscGetTime.o

PetscGetCPUTime: PetscGetCPUTime.o 
	-${CLINKER} -o PetscGetCPUTime PetscGetCPUTime.o ${PETSC_LIB}
	${RM} -f PetscGetCPUTime.o

MPI_Wtime: MPI_Wtime.o 
	-${CLINKER} -o MPI_Wtime MPI_Wtime.o ${PETSC_LIB}
	${RM} -f MPI_Wtime.o

PLogEvent: PLogEvent.o 
	-${CLINKER} -o PLogEvent PLogEvent.o ${PETSC_LIB}
	${RM} -f PLogEvent.o

PetscMalloc: PetscMalloc.o 
	-${CLINKER} -o PetscMalloc PetscMalloc.o ${PETSC_LIB}
	${RM} -f PetscMalloc.o

PetscMemcpy: PetscMemcpy.o 
	-${CLINKER} -o PetscMemcpy PetscMemcpy.o ${PETSC_LIB}
	${RM} -f PetscMemcpy.o

PetscMemzero: PetscMemzero.o 
	-${CLINKER} -o PetscMemzero PetscMemzero.o ${PETSC_LIB}
	${RM} -f PetscMemzero.o

PetscMemcmp: PetscMemcmp.o 
	-${CLINKER} -o PetscMemcmp PetscMemcmp.o ${PETSC_LIB}
	${RM} -f PetscMemcmp.o

Index: Index.o 
	-${CLINKER} -o Index Index.o ${PETSC_LIB}
	${RM} -f Index.o

PetscVecNorm: PetscVecNorm.o 
	-${CLINKER} -o PetscVecNorm PetscVecNorm.o ${PETSC_LIB}
	${RM} -f PetscVecNorm.o

sizeof: sizeof.o 
	-${CLINKER} -o sizeof sizeof.o ${PETSC_LIB}
	${RM} -f sizeof.o

test: ${TESTS}

runtest:
	-@echo "Time Taken by some PETSc routines are as follows:"
	-@echo "------------------------------------------------"
	-@${MPIEXEC} -n 1 ./PetscTime
	-@${MPIEXEC} -n 1 ./PetscGetTime
	-@${MPIEXEC} -n 1 ./PetscGetCPUTime
	-@${MPIEXEC} -n 1 ./MPI_Wtime
	-@echo " "
	-@echo "PLogEventBegin and PLogEventEnd together with options"
	-@echo "------------------------------------------------"
	-@${MPIEXEC} -n 1 ./PLogEvent              > /dev/null
	-@${MPIEXEC} -n 1 ./PLogEvent -log_view > /dev/null
	-@${MPIEXEC} -n 1 ./PLogEvent -log_mpe     > /dev/null
	-@echo " "
	-@echo "PetscMalloc and PetscFree together  with options"
	-@echo "------------------------------------------------"
	-@${MPIEXEC} -n 1 ./PetscMalloc
	-@${MPIEXEC} -n 1 ./PetscMalloc -malloc
	-@echo " "
	-@echo "Memory Operations "
	-@echo "------------------------------------------------"
	-@${MPIEXEC} -n 1 ./PetscMemcpy
	-@${MPIEXEC} -n 1 ./PetscMemzero
	-@${MPIEXEC} -n 1 ./PetscMemcmp
	-@echo " "
	-@echo "Memory Operations "
	-@echo "------------------------------------------------"
	-@${MPIEXEC} -n 1 ./Index
	-@echo " "
	-@echo "Datatype Sizes "
	-@echo "------------------------------------------------"
	-@${MPIEXEC} -n 1 ./sizeof
	-@echo "------------------------------------------------"
