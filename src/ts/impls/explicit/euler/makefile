# $Id: makefile,v 1.8 1997/09/10 19:04:51 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__DIR__='"${LOCDIR}"' 
SOURCEC  = euler.c
SOURCEF  =
SOURCEH  = 
OBJSC    = euler.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/impls/explicit/euler/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

