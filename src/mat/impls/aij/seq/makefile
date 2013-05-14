
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = aij.c aijfact.c ij.c fdaij.c \
	   matmatmult.c symtranspose.c matptap.c matrart.c inode.c inode2.c matmatmatmult.c
SOURCEF  =
SOURCEH  = aij.h
LIBBASE  = libpetscmat
DIRS     = superlu umfpack essl lusol matlab csrperm crl bas ftn-kernels seqcusp seqviennacl \
           cholmod seqcusparse
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
