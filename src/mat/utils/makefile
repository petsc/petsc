# $Id: makefile,v 1.35 1998/04/23 23:29:13 bsmith Exp balay $

ALL: lib

CFLAGS   =  ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = convert.c stash.c axpy.c zerodiag.c matio.c matioall.c \
           conaij.c conbdiag.c getcolv.c gcreate.c
SOURCEF  =
SOURCEH  =
OBJSC    = convert.o stash.o axpy.o zerodiag.o matio.o matioall.o \
           conaij.o conbdiag.o getcolv.o gcreate.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = 2
LOCDIR   = src/mat/utils/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
