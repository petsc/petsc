# $Id: makefile,v 1.30 1997/09/10 18:53:40 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = bvec2.c bvec1.c dvec2.c
SOURCEF  =
SOURCEH  = 
OBJSC    = bvec2.o  bvec1.o dvec2.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1
LOCDIR   = src/vec/impls/seq/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

