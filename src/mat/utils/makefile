# $Id: makefile,v 1.41 1999/04/16 16:07:54 bsmith Exp bsmith $

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = convert.c matstash.c axpy.c zerodiag.c matio.c matioall.c \
           conaij.c conbdiag.c getcolv.c gcreate.c dlregis.c
SOURCEF  =
SOURCEH  =
OBJSC    = convert.o matstash.o axpy.o zerodiag.o matio.o matioall.o \
           conaij.o conbdiag.o getcolv.o gcreate.o dlregis.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = Mat
LOCDIR   = src/mat/utils/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test

