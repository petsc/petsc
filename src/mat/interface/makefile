# $Id: makefile,v 1.39 2000/08/17 04:51:22 bsmith Exp bsmith $ 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = matrix.c mhas.c matreg.c matregis.c matnull.c
SOURCEF  =
SOURCEH  = 
OBJSC    = matrix.o mhas.o matreg.o matregis.o matnull.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = Mat
LOCDIR   = src/mat/interface/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
