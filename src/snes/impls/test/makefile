# $Id: makefile,v 1.19 1997/09/10 19:02:37 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = snestest.c
SOURCEF  =
SOURCEH  = 
OBJSC    = snestest.o
OBJSF    =
LIBBASE  = libpetscsnes
MANSEC   = n
LOCDIR   = src/snes/impls/test/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

