
CFLAGS	  =
FFLAGS    =
CPPFLAGS  =
FPPFLAGS  =
LOCDIR    = src/ksp/ksp/examples/tests/benchmarkscatters
EXAMPLESC       = ex1.c ex2.c

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_KSP_LIB}
	${RM} ex1.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_KSP_LIB}
	${RM} ex2.o

stencil_mpi_shmem: stencil_mpi_shmem.o  chkopts
	-${CLINKER} -o stencil_mpi_shmem stencil_mpi_shmem.o  ${PETSC_KSP_LIB}
	${RM} stencil_mpi_shmem.o

stencil_mpi_shmem1: stencil_mpi_shmem1.o  chkopts
	-${CLINKER} -o stencil_mpi_shmem1 stencil_mpi_shmem1.o  ${PETSC_KSP_LIB}
	${RM} stencil_mpi_shmem1.o

TESTEXAMPLES_C		       = ex1.PETSc ex1.rm
include ${PETSC_DIR}/lib/petsc/conf/test
