# $Id: makefile,v 1.34 1998/03/19 02:05:03 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = pbvec.c pdvec.c pvec2.c
SOURCEF  =
SOURCEH  = pvecimpl.h 
OBJSC    = pbvec.o pdvec.o pvec2.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1
LOCDIR   = src/vec/impls/mpi/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

