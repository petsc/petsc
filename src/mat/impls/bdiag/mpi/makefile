# $Id: makefile,v 1.19 1997/09/10 18:48:24 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = mmbdiag.c mpibdiag.c mpibdpc.c
SOURCEF  =
SOURCEH  = mpibdiag.h
OBJSC    = mmbdiag.o mpibdiag.o mpibdpc.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2
LOCDIR   = src/mat/impls/bdiag/mpi/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
