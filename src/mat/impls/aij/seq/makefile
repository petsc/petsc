
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = aij.c aijfact.c ij.c fdaij.c \
	   matmatmult.c symtranspose.c matptap.c matpapt.c inode.c inode2.c
SOURCEF  = 
SOURCEH  = aij.h
OBJSC    = aij.o aijfact.o ij.o fdaij.o \
	   matmatmult.o symtranspose.o matptap.o matpapt.o inode.o inode2.o
OBJSF    = 
LIBBASE  = libpetscmat
DIRS     = spooles superlu umfpack essl lusol matlab csrperm crl bas ftn-kernels
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
