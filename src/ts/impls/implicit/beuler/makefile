# $Id: makefile,v 1.10 1998/03/19 02:11:18 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = beuler.c
SOURCEF  =
SOURCEH  = 
OBJSC    = beuler.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/impls/implicit/beuler/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

