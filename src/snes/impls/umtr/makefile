# $Id: makefile,v 1.17 1997/09/10 19:02:40 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = umtr.c
SOURCEF  =
SOURCEH  = umtr.h
OBJSC    = umtr.o
OBJSF    =
LIBBASE  = libpetscsnes
MANSEC   = n
LOCDIR   = src/snes/impls/umtr/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

