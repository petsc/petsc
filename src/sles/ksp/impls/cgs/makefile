# $Id: makefile,v 1.25 1998/03/19 02:05:28 balay Exp balay $

ALL: lib 

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = cgs.c
SOURCEF  =
OBJSC    = cgs.o
OBJSF    = 
LIBBASE  = libpetscsles
LOCDIR   = src/ksp/impls/cgs/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
