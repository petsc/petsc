# $Id: makefile,v 1.26 1997/09/10 18:52:18 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = lu.c
SOURCEF  =
SOURCEH  = 
OBJSC    = lu.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4
LOCDIR   = src/pc/impls/lu/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
