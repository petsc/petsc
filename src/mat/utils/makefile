# $Id: makefile,v 1.39 1999/02/02 21:27:20 balay Exp balay $

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = convert.c matstash.c axpy.c zerodiag.c matio.c matioall.c \
           conaij.c conbdiag.c getcolv.c gcreate.c
SOURCEF  =
SOURCEH  =
OBJSC    = convert.o matstash.o axpy.o zerodiag.o matio.o matioall.o \
           conaij.o conbdiag.o getcolv.o gcreate.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = Mat
LOCDIR   = src/mat/utils/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
