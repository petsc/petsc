
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = aij.c aijfact.c ij.c fdaij.c unary.c ilut.c \
	   matmatmult.c symtranspose.c matptap.c matpapt.c inode.c inode2.c
SOURCEF  = 
SOURCEH  = aij.h
OBJSC    = aij.o aijfact.o ij.o fdaij.o unary.o ilut.o \
	   matmatmult.o symtranspose.o matptap.o matpapt.o inode.o inode2.o
OBJSF    = 
LIBBASE  = libpetscmat
DIRS     = spooles superlu umfpack essl lusol matlab csrperm crl
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
