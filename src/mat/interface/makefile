# $Id: makefile,v 1.38 2000/08/01 20:01:52 bsmith Exp bsmith $ 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = matrix.c mhas.c matcnv.c mgtype.c matreg.c matregis.c matnull.c
SOURCEF  =
SOURCEH  = 
OBJSC    = matrix.o mhas.o matcnv.o mgtype.o matreg.o matregis.o matnull.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = Mat
LOCDIR   = src/mat/interface/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
