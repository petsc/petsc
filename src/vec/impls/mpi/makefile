# $Id: makefile,v 1.35 1998/04/23 23:39:00 balay Exp bsmith $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = pbvec.c pdvec.c pvec2.c pmap.c
SOURCEF  =
SOURCEH  = pvecimpl.h 
OBJSC    = pbvec.o pdvec.o pvec2.o pmap.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1
LOCDIR   = src/vec/impls/mpi/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

