
ALL: 

CFLAGS	      =
FFLAGS	      =	 
CPPFLAGS      =
FPPFLAGS      =
LOCDIR        = src/benchmarks/
EXAMPLESC     = PetscTime.c PetscGetTime.c MPI_Wtime.c PLogEvent.c PetscMalloc.c \
		PetscMemcpy.c PetscMemzero.c PetscMemcmp.c Index.c PetscVecNorm.c \
		PetscGetCPUTime.c
EXAMPLESF     = 
TESTS         = PetscTime PetscGetTime MPI_Wtime PLogEvent PetscMalloc \
		PetscMemcpy PetscMemzero PetscMemcmp Index PetscVecNorm \
		PetscGetCPUTime sizeof
MANSEC        = Sys

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

PetscTime: PetscTime.o  chkopts 
	-${CLINKER} -o PetscTime PetscTime.o ${PETSC_LIB}
	${RM} -f PetscTime.o

PetscGetTime: PetscGetTime.o  chkopts 
	-${CLINKER} -o PetscGetTime PetscGetTime.o ${PETSC_LIB}
	${RM} -f PetscGetTime.o

PetscGetCPUTime: PetscGetCPUTime.o  chkopts 
	-${CLINKER} -o PetscGetCPUTime PetscGetCPUTime.o ${PETSC_LIB}
	${RM} -f PetscGetCPUTime.o

MPI_Wtime: MPI_Wtime.o  chkopts 
	-${CLINKER} -o MPI_Wtime MPI_Wtime.o ${PETSC_LIB}
	${RM} -f MPI_Wtime.o

PLogEvent: PLogEvent.o  chkopts 
	-${CLINKER} -o PLogEvent PLogEvent.o ${PETSC_LIB}
	${RM} -f PLogEvent.o

PetscMalloc: PetscMalloc.o  chkopts 
	-${CLINKER} -o PetscMalloc PetscMalloc.o ${PETSC_LIB}
	${RM} -f PetscMalloc.o

PetscMemcpy: PetscMemcpy.o  chkopts 
	-${CLINKER} -o PetscMemcpy PetscMemcpy.o ${PETSC_LIB}
	${RM} -f PetscMemcpy.o

PetscMemzero: PetscMemzero.o  chkopts 
	-${CLINKER} -o PetscMemzero PetscMemzero.o ${PETSC_LIB}
	${RM} -f PetscMemzero.o

PetscMemcmp: PetscMemcmp.o  chkopts 
	-${CLINKER} -o PetscMemcmp PetscMemcmp.o ${PETSC_LIB}
	${RM} -f PetscMemcmp.o

Index: Index.o  chkopts 
	-${CLINKER} -o Index Index.o ${PETSC_LIB}
	${RM} -f Index.o

PetscVecNorm: PetscVecNorm.o  chkopts 
	-${CLINKER} -o PetscVecNorm PetscVecNorm.o ${PETSC_LIB}
	${RM} -f PetscVecNorm.o

sizeof: sizeof.o  chkopts 
	-${CLINKER} -o sizeof sizeof.o ${PETSC_LIB}
	${RM} -f sizeof.o

test: ${TESTS}

runtest: 
	-@echo "Time Taken by some PETSc routines are as follows:"
	-@echo "------------------------------------------------"
	-@${MPIRUN} -np 1 ./PetscTime
	-@${MPIRUN} -np 1 ./PetscGetTime
	-@${MPIRUN} -np 1 ./PetscGetCPUTime
	-@${MPIRUN} -np 1 ./MPI_Wtime
	-@echo " "
	-@echo "PLogEventBegin and PLogEventEnd together with options"
	-@echo "------------------------------------------------"
	-@${MPIRUN} -np 1 ./PLogEvent              > /dev/null
	-@${MPIRUN} -np 1 ./PLogEvent -log_summary > /dev/null
	-@${MPIRUN} -np 1 ./PLogEvent -log_mpe     > /dev/null
	-@echo " "
	-@echo "PetscMalloc and PetscFree together  with options"
	-@echo "------------------------------------------------"
	-@${MPIRUN} -np 1 ./PetscMalloc
	-@${MPIRUN} -np 1 ./PetscMalloc -malloc
	-@echo " "
	-@echo "Memory Operations "
	-@echo "------------------------------------------------"
	-@${MPIRUN} -np 1 ./PetscMemcpy
	-@${MPIRUN} -np 1 ./PetscMemzero
	-@${MPIRUN} -np 1 ./PetscMemcmp
	-@echo " "
	-@echo "Memory Operations "
	-@echo "------------------------------------------------"
	-@${MPIRUN} -np 1 ./Index
	-@echo " "
	-@echo "Datatype Sizes "
	-@echo "------------------------------------------------"
	-@${MPIRUN} -np 1 ./sizeof
	-@echo "------------------------------------------------"
