# $Id: makefile,v 1.23 1998/03/19 02:12:59 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = ls.c
SOURCEF  =
SOURCEH  = ls.h
OBJSC    = ls.o
OBJSF    =
LIBBASE  = libpetscsnes
MANSEC   = n
LOCDIR   = src/snes/impls/ls/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

