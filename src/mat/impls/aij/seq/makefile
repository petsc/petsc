
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = aij.c aijfact.c ij.c fdaij.c \
	   matmatmult.c symtranspose.c matptap.c matrart.c inode.c inode2.c matmatmatmult.c \
           mattransposematmult.c
SOURCEF  =
SOURCEH  = aij.h
LIBBASE  = libpetscmat
DIRS     = superlu umfpack essl lusol matlab csrperm aijmkl crl bas ftn-kernels seqcusp seqviennacl seqviennaclcuda \
           cholmod seqcusparse klu mkl_pardiso
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
