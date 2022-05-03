-include ../../../../../petscdir.mk

SOURCEC  = aij.c aijfact.c ij.c fdaij.c matmatmult.c symtranspose.c matptap.c matrart.c inode.c inode2.c matmatmatmult.c mattransposematmult.c aijhdf5.c
SOURCEF  =
SOURCEH  = aij.h
LIBBASE  = libpetscmat
DIRS     = superlu umfpack essl lusol matlab aijperm aijsell aijmkl crl bas ftn-kernels seqviennacl seqviennaclcuda cholmod seqcusparse klu mkl_pardiso kokkos spqr
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
