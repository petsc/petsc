# $Id: makefile,v 1.50 2001/08/22 18:00:26 balay Exp $ 

ALL: lib

CFLAGS   = ${MATLAB_ENGINE_INCLUDE} ${DSCPACK_INCLUDE}
FFLAGS   =
SOURCEC  = aij.c aijfact.c ij.c essl.c aijnode.c fdaij.c unary.c ilut.c aijmatlab.c \
	   lusol.c dsc.c matmatmult.c symtranspose.c matptap.c matpapt.c
SOURCEF  = 
SOURCEH  = aij.h
OBJSC    = aij.o aijfact.o ij.o essl.o aijnode.o fdaij.o unary.o ilut.o aijmatlab.o \
	   lusol.o dsc.o matmatmult.o symtranspose.o matptap.o matpapt.o
OBJSF    = 
LIBBASE  = libpetscmat
DIRS     = spooles superlu umfpack dxml
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
