# $Id: makefile,v 1.44 2000/08/17 04:51:27 bsmith Exp bsmith $ 

ALL: lib

CFLAGS   = ${MATLAB_INCLUDE}
FFLAGS   =
SOURCEC  = aij.c aijfact.c ij.c essl.c dxml.c aijnode.c fdaij.c unary.c ilut.c aijmatlab.c lusol.c
SOURCEF  = ${PETSC_LUSOL_SOURCE_F}
SOURCEH  = aij.h
OBJSC    = aij.o aijfact.o ij.o essl.o dxml.o aijnode.o fdaij.o unary.o ilut.o aijmatlab.o lusol.o
OBJSF    = ${PETSC_LUSOL_SOURCE_O}
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
