# $Id: makefile,v 1.24 1997/09/10 18:51:48 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = sor.c
SOURCEF  =
SOURCEH  = 
OBJSC    = sor.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4
LOCDIR   = src/pc/impls/sor/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
