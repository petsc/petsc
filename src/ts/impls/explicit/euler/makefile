# $Id: makefile,v 1.9 1998/03/19 02:11:14 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__DIR__='"${LOCDIR}"' 
SOURCEC  = euler.c
SOURCEF  =
SOURCEH  = 
OBJSC    = euler.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/impls/explicit/euler/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

