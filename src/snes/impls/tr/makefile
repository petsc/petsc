# $Id: makefile,v 1.21 1997/09/10 19:02:25 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = tr.c
SOURCEF  =
SOURCEH  = tr.h
OBJSC    = tr.o
OBJSF    =
LIBBASE  = libpetscsnes
MANSEC   = n
LOCDIR   = src/snes/impls/tr/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

