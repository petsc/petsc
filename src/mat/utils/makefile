# $Id: makefile,v 1.42 2000/08/17 04:51:56 bsmith Exp bsmith $

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = convert.c matstash.c axpy.c zerodiag.c matio.c matioall.c \
           getcolv.c gcreate.c dlregis.c
SOURCEF  =
SOURCEH  =
OBJSC    = convert.o matstash.o axpy.o zerodiag.o matio.o matioall.o \
           getcolv.o gcreate.o dlregis.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = Mat
LOCDIR   = src/mat/utils/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test

