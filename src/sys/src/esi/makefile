# $Id: makefile,v 1.6 2001/08/24 21:03:07 bsmith Exp bsmith $

ALL: lib

CFLAGS   = ${ESI_INCLUDE} -I../include
FFLAGS   =
SOURCEC  = object.c map.c vector.c matrix.c
OBJSC    = object.o map.o vector.o matrix.o
SOURCEH  = ../include/petsc/map.h ../include/petsc/vector.h ../include/petsc/matrix.h
LIBBASE  = libpetscvec
MANSEC   = 
LOCDIR   = src/contrib/esi/src


include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


