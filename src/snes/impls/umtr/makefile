# $Id: makefile,v 1.18 1998/03/19 02:13:11 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = umtr.c
SOURCEF  =
SOURCEH  = umtr.h
OBJSC    = umtr.o
OBJSF    =
LIBBASE  = libpetscsnes
MANSEC   = n
LOCDIR   = src/snes/impls/umtr/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

