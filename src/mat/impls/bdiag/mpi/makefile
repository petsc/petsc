# $Id: makefile,v 1.20 1998/03/19 02:07:55 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = mmbdiag.c mpibdiag.c mpibdpc.c
SOURCEF  =
SOURCEH  = mpibdiag.h
OBJSC    = mmbdiag.o mpibdiag.o mpibdpc.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2
LOCDIR   = src/mat/impls/bdiag/mpi/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
