# $Id: makefile,v 1.9 1997/09/10 19:04:40 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = posindep.c
SOURCEF  =
SOURCEH  = 
OBJSC    = posindep.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/impls/pseudo/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

