# $Id: makefile,v 1.10 1998/03/19 02:11:25 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = posindep.c
SOURCEF  =
SOURCEH  = 
OBJSC    = posindep.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/impls/pseudo/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

