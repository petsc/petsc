# $Id: makefile,v 1.19 1997/08/22 15:17:44 bsmith Exp balay $

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF}  -D__SDIR__='"${LOCDIR}"'
SOURCEC  = view.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = view.o 
OBJSF    =
LIBBASE  = libpetscsys
MANSEC   = v
LOCDIR   = src/viewer/interface/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

