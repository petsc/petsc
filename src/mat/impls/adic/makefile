# $Id: makefile,v 1.47 2001/01/16 18:17:28 balay Exp bsmith $ 

ALL: lib

CFLAGS   = ${MATLAB_INCLUDE} ${DSCPACK_INCLUDE} ${ADIC_INCLUDE}
FFLAGS   =
SOURCEC  = aij.c aijfact.c ij.c essl.c dxml.c aijnode.c fdaij.c unary.c ilut.c aijmatlab.c lusol.c dsc.c
SOURCEF  = 
SOURCEH  = aij.h
OBJSC    = aij.o aijfact.o ij.o essl.o dxml.o aijnode.o fdaij.o unary.o ilut.o aijmatlab.o lusol.o dsc.o
OBJSF    = 
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
