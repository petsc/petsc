# $Id: makefile,v 1.22 1997/09/10 19:02:32 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = ls.c
SOURCEF  =
SOURCEH  = ls.h
OBJSC    = ls.o
OBJSF    =
LIBBASE  = libpetscsnes
MANSEC   = n
LOCDIR   = src/snes/impls/ls/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

