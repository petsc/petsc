# $Id: makefile,v 1.43 2001/08/22 18:02:32 balay Exp $ 

ALL: lib

CFLAGS   = ${MATLAB_ENGINE_INCLUDE}
FFLAGS   =
SOURCEC  = da2.c da1.c da3.c daghost.c dacorn.c dagtol.c daltol.c \
           daltog.c daindex.c dascatter.c dadestroy.c dalocal.c \
           dadist.c daview.c dasub.c gr1.c gr2.c daload.c dagtona.c dainterp.c dapf.c dagetarray.c daglenn.c
OBJSC    = da2.o da1.o da3.o daghost.o dacorn.o dagtol.o daltol.o \
           daltog.o daindex.o dascatter.o dadestroy.o dalocal.o \
           dadist.o daview.o dasub.o gr1.o gr2.o daload.o dagtona.o dainterp.o dapf.o dagetarray.o daglenn.o
LIBBASE  = libpetscdm
MANSEC   = DA
LOCDIR   = src/dm/da/src/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
