# $Id: makefile,v 1.43 2000/10/24 20:26:14 bsmith Exp balay $

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

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

