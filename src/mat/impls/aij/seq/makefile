# $Id: makefile,v 1.50 2001/08/22 18:00:26 balay Exp $ 

ALL: lib

CFLAGS   = ${MATLAB_ENGINE_INCLUDE} ${DSCPACK_INCLUDE} ${SUPERLU_INCLUDE} ${SPOOLES_INCLUDE} ${UMFPACK_INCLUDE}
FFLAGS   =
SOURCEC  = aij.c aijfact.c ij.c essl.c dxml.c aijnode.c fdaij.c unary.c ilut.c aijmatlab.c \
	   lusol.c dsc.c superlu.c aijspooles.c spoolesOptions.c spooles.c umfpack.c matmatmult.c \
	   symtranspose.c matptap.c matpapt.c
SOURCEF  = 
SOURCEH  = aij.h spooles.h
OBJSC    = aij.o aijfact.o ij.o essl.o dxml.o aijnode.o fdaij.o unary.o ilut.o aijmatlab.o \
	   lusol.o dsc.o superlu.o aijspooles.o spoolesOptions.o spooles.o umfpack.o matmatmult.o \
	   symtranspose.o matptap.o matpapt.o
OBJSF    = 
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
