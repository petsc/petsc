# $Id: makefile,v 1.44 2001/08/22 18:01:10 balay Exp $

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = convert.c matstash.c axpy.c zerodiag.c matio.c matioall.c \
           getcolv.c gcreate.c freespace.c
SOURCEF  =
SOURCEH  = freespace.h
OBJSC    = convert.o matstash.o axpy.o zerodiag.o matio.o matioall.o \
           getcolv.o gcreate.o freespace.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = Mat
LOCDIR   = src/mat/utils/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

