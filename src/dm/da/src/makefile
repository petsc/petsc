# $Id: makefile,v 1.28 1998/06/15 15:47:04 balay Exp bsmith $ 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = da2.c da1.c da3.c daghost.c dacorn.c dagtol.c daltol.c \
           daltog.c daindex.c dascatter.c dadestroy.c dalocal.c \
           dadist.c daview.c dasub.c gr1.c
OBJSC    = da2.o da1.o da3.o daghost.o dacorn.o dagtol.o daltol.o \
           daltog.o daindex.o dascatter.o dadestroy.o dalocal.o \
           dadist.o daview.o dasub.o gr1.o
LIBBASE  = libpetscvec
MANSEC   = 6
LOCDIR   = src/da/src/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
