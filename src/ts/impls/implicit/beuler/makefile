# $Id: makefile,v 1.9 1997/09/10 19:04:57 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = beuler.c
SOURCEF  =
SOURCEH  = 
OBJSC    = beuler.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/impls/implicit/beuler/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

