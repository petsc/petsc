# $Id: makefile,v 1.28 1997/09/10 18:51:57 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = bjacobi.c
SOURCEF  =
SOURCEH  = bjacobi.h
OBJSC    = bjacobi.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4
LOCDIR   = src/pc/impls/bjacobi/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
