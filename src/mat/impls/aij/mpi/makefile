-include ../../../../../petscdir.mk

SOURCEC	 = mpiaij.c mmaij.c mpiaijpc.c mpiov.c fdmpiaij.c mpiptap.c mpimatmatmult.c mpb_aij.c mpimatmatmatmult.c mpimattransposematmult.c
SOURCEF	 =
SOURCEH	 = mpiaij.h
LIBBASE	 = libpetscmat
DIRS	   = superlu_dist mumps aijperm aijmkl aijsell crl pastix mpicusparse mpiviennacl mpiviennaclcuda clique mkl_cpardiso strumpack kokkos
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
