# $Id: makefile,v 1.24 1997/08/22 15:11:22 bsmith Exp balay $

ALL: lib 

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = cgs.c
SOURCEF  =
OBJSC    = cgs.o
OBJSF    = 
LIBBASE  = libpetscsles
LOCDIR   = src/ksp/impls/cgs/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
